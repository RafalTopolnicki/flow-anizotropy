// ---------------------------------------------------------------------------
// lbm3d-perm — permeability tensor of a 3D triply periodic porous cell
//
// The 3D counterpart of LMB2d/lbm2d-perm.  Darcy's law for a periodic cell is
// q_i = (K_ij / mu) F_j, so the full 3x3 tensor needs three solves of the same
// structure, one per forcing axis, each contributing one column of K:
//
//     run x:  k_xx = mu q_x / F,  k_yx = mu q_y / F,  k_zx = mu q_z / F
//     run y:  k_xy, k_yy, k_zy
//     run z:  k_xz, k_yz, k_zz
//
// All three run in one invocation so the structure is read once and the three
// reciprocity residuals — K = K^T is exact for Stokes flow — are available per
// sample as free error bars on the off-diagonals.  In 2D that gave one free
// error bar; here it gives three, one per off-diagonal target.
//
// The convergence machinery is ported unchanged in spirit from the 2D solver,
// where it was arrived at by measurement (NOTES.md section 3):
//
//   * the test is on *block means* of the domain-averaged flux, not on
//     instantaneous samples.  Past the transient the residuals stop decaying
//     and fluctuate; an instantaneous test never accumulates consecutive
//     passes and runs to the step cap on a solution that settled long ago.
//   * every component is tested against its own magnitude.  The transverse
//     components are heavily cancelling sums and converge much later than the
//     driven one, so a single scalar residual is meaningless.
//   * the tolerance is the looser of a relative bound and k-sigma of the block
//     mean's own standard error, which makes it self-calibrating.
//   * --hold consecutive passes are required, then q is block-averaged over a
//     further --avg-steps rather than read off at the stopping step.
//
// Two deliberate departures from the 2D code, neither of which has a legacy
// dataset to stay compatible with:
//
//   0. Guo (2002) forcing, with the physical velocity carrying half the body
//      force, in place of the 2D code's equilibrium-velocity shift and raw
//      moment sum(f e)/rho.  Measured on the analytic Poiseuille slit, the 2D
//      convention leaves the reported velocity a uniform F/4 below the exact
//      profile — harmless in 2D, where it was kept for comparability with the
//      earlier surveys and cancels from every off-diagonal, but it also makes
//      a *blocked* direction report a small negative permeability instead of
//      zero.  With Guo forcing the solver reproduces the analytic slit to
//      ~1e-12 and a blocked direction returns exactly 0.
//   1. Distributions are double, not float.  NOTES 3.2 measured the 2D
//      residual flooring at 1e-3..1e-4 because U/V are float and BGK carries a
//      persistent fluctuation.  At 80^3 the memory cost of double is ~156 MB
//      per process, which is affordable, and it removes that floor.
//   2. q_i = mean(u_i), the standard Darcy superficial velocity, rather than
//      the 2D code's sum(u_i)/(L*L0^2) with L0 = 4.  The 2D form was kept only
//      so k stayed comparable with the earlier k0/k2/deeppore surveys; there is
//      no such constraint here.  Consequently **3D k is not comparable in
//      magnitude with 2D k** (they differ by L0^2 = 16, and by the 1% between
//      the 2D code's mu = 0.33(tau-0.5) and the exact cs^2 (tau-0.5) used
//      here), and by the forcing scheme in item 0.  Every ratio, anisotropy and
//      principal direction is unaffected.
//
// Structure input is the .raw format written by scripts/structure_io3d.py:
// three little-endian int32 (nx, ny, nz) then nx*ny*nz uint8, 1 = solid, with
// x varying fastest.  No image library is needed, so this build has no
// external dependencies at all.
//
//   Usage: lbm3d-perm <structure.raw> <results.csv> [options]
// ---------------------------------------------------------------------------

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

using namespace std;

// --- D3Q19 -----------------------------------------------------------------
static const int Q = 19;
static const int ex[Q] = {0, 1,-1, 0, 0, 0, 0, 1,-1, 1,-1, 1,-1, 1,-1, 0, 0, 0, 0};
static const int ey[Q] = {0, 0, 0, 1,-1, 0, 0, 1,-1,-1, 1, 0, 0, 0, 0, 1,-1, 1,-1};
static const int ez[Q] = {0, 0, 0, 0, 0, 1,-1, 0, 0, 0, 0, 1,-1,-1, 1, 1,-1,-1, 1};
static const double wq[Q] = {
    1.0/3.0,
    1.0/18.0, 1.0/18.0, 1.0/18.0, 1.0/18.0, 1.0/18.0, 1.0/18.0,
    1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0,
    1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0};
static int opp[Q];

static void build_opp()
{
    for(int k = 0; k < Q; k++)
        for(int m = 0; m < Q; m++)
            if(ex[m] == -ex[k] && ey[m] == -ey[k] && ez[m] == -ez[k]) { opp[k] = m; break; }
}

static const double TAU     = 1.0;
static const double CS2     = 1.0/3.0;
static const double MU      = CS2 * (TAU - 0.5);   // kinematic viscosity, rho0 = 1
static const double FX_BASE = 2.5e-07;             // body force at force factor 1

// --- configuration ---------------------------------------------------------
struct Config
{
    double ff         = 1.0;
    double eps        = 1e-3;
    long   lag        = 2000;
    int    hold       = 3;
    long   min_steps  = 4000;
    long   max_steps  = 200000;
    long   avg_steps  = 8000;
    long   avg_sample = 25;
    double qfloor     = 1e-2;
    double ksigma     = 3.0;
    string velocity_prefix;
    bool   quiet      = false;
    bool   validate   = false;
};

struct Solve
{
    double q[3]  = {0,0,0};
    double se[3] = {0,0,0};
    long   steps = 0;
    int    converged = 0;
};

static void die(const string &msg)
{
    cerr << "lbm3d-perm: " << msg << endl;
    exit(2);
}

// --- lattice state ---------------------------------------------------------
static int  NX, NY, NZ;
static long NN;
static vector<uint8_t> solid;
static vector<double>  f0, f1;                 // NN*Q
static vector<double>  U, V, W, RHO;           // NN
static double Fx, Fy, Fz;
static long   n_pore;

static inline long idx(int x, int y, int z) { return (long(z)*NY + y)*NX + x; }

// Precomputed wrapped neighbour coordinates for the *pull* direction (-e_k).
// Tiny (Q * N per axis) so they stay in cache, and they keep the modulo out of
// the inner loop.
static vector<int> pmx, pmy, pmz;

static void build_neighbour_tables()
{
    pmx.resize(size_t(Q)*NX); pmy.resize(size_t(Q)*NY); pmz.resize(size_t(Q)*NZ);
    for(int k = 0; k < Q; k++)
    {
        for(int x = 0; x < NX; x++) pmx[size_t(k)*NX + x] = ((x - ex[k]) % NX + NX) % NX;
        for(int y = 0; y < NY; y++) pmy[size_t(k)*NY + y] = ((y - ey[k]) % NY + NY) % NY;
        for(int z = 0; z < NZ; z++) pmz[size_t(k)*NZ + z] = ((z - ez[k]) % NZ + NZ) % NZ;
    }
}

static void read_structure(const string &path)
{
    ifstream in(path.c_str(), ios::binary);
    if(!in) die("cannot open " + path);
    int32_t hdr[3];
    in.read(reinterpret_cast<char*>(hdr), 12);
    if(!in) die("cannot read header of " + path);
    NX = hdr[0]; NY = hdr[1]; NZ = hdr[2];
    if(NX <= 0 || NY <= 0 || NZ <= 0) die("bad dimensions in " + path);
    NN = long(NX) * NY * NZ;
    solid.resize(NN);
    in.read(reinterpret_cast<char*>(solid.data()), NN);
    if(!in) die("truncated payload in " + path);
    in.peek();
    if(!in.eof()) die("trailing bytes in " + path);

    n_pore = 0;
    for(long i = 0; i < NN; i++) { if(solid[i] > 1) die("non-binary voxel"); if(!solid[i]) n_pore++; }
    if(n_pore == 0) die("structure has no pore space");
}

static void allocate()
{
    f0.assign(NN*Q, 0.0); f1.assign(NN*Q, 0.0);
    U.assign(NN, 0.0); V.assign(NN, 0.0); W.assign(NN, 0.0); RHO.assign(NN, 1.0);
}

// Reset to a uniform quiescent state, so each forcing direction starts from
// the same place the first one did.  `solid` is deliberately left alone.
static void reset_distributions()
{
    for(long i = 0; i < NN; i++)
        for(int k = 0; k < Q; k++) f0[i*Q+k] = f1[i*Q+k] = wq[k];
    fill(U.begin(), U.end(), 0.0);
    fill(V.begin(), V.end(), 0.0);
    fill(W.begin(), W.end(), 0.0);
    fill(RHO.begin(), RHO.end(), 1.0);
}

static void macro_collide()
{
    for(long i = 0; i < NN; i++)
    {
        if(solid[i]) continue;                 // U,V,W stay exactly 0 in solid
        double *f = &f0[i*Q];
        double r = 0, ux = 0, uy = 0, uz = 0;
        for(int k = 0; k < Q; k++)
        {
            r += f[k];
            ux += f[k]*ex[k]; uy += f[k]*ey[k]; uz += f[k]*ez[k];
        }
        // Guo forcing: the physical velocity carries half the body force.
        // Macro and collision are fused into this one pass: both are node-local
        // on f0, so splitting them cost a second full sweep of an array far
        // larger than cache for no benefit.
        RHO[i] = r;
        ux = U[i] = (ux + 0.5*Fx)/r;
        uy = V[i] = (uy + 0.5*Fy)/r;
        uz = W[i] = (uz + 0.5*Fz)/r;

        const double u2 = ux*ux + uy*uy + uz*uz;
        const double pref = 1.0 - 0.5/TAU;
        for(int k = 0; k < Q; k++)
        {
            const double eu = ex[k]*ux + ey[k]*uy + ez[k]*uz;
            const double feq = wq[k]*r*(1.0 + 3.0*eu + 4.5*eu*eu - 1.5*u2);
            // Guo et al. (2002) source term, with cs^2 = 1/3:
            //   S_k = w_k (1 - 1/2tau) [ 3(e_k - u) + 9(e_k.u) e_k ] . F
            const double sk = wq[k] * pref *
                ( 3.0*((ex[k]-ux)*Fx + (ey[k]-uy)*Fy + (ez[k]-uz)*Fz)
                + 9.0*eu*(ex[k]*Fx + ey[k]*Fy + ez[k]*Fz) );
            f[k] += (feq - f[k]) / TAU + sk;
        }
    }
}

// Halfway bounce-back on links, periodic in all three axes, as a *pull*:
//
//     p = i - e_k ;  f1[i][k] = solid[p] ? f0[i][opp[k]] : f0[p][k]
//
// which is exactly equivalent to the push form (a push writes f1[i][k] either
// from the node at i - e_k streaming along k, or from node i itself bouncing
// direction opp[k] back when i - e_k is solid).  Pull is worth the rewrite:
// every entry of f1 is written exactly once, so the full 78 MB memset the push
// form needed each step disappears, and the writes become contiguous instead of
// scattered.
static void stream()
{
    for(int z = 0; z < NZ; z++)
    for(int y = 0; y < NY; y++)
    for(int x = 0; x < NX; x++)
    {
        const long i = idx(x, y, z);
        double *dst = &f1[i*Q];
        if(solid[i]) { for(int k = 0; k < Q; k++) dst[k] = 0.0; continue; }
        for(int k = 0; k < Q; k++)
        {
            const long p = idx(pmx[size_t(k)*NX + x],
                               pmy[size_t(k)*NY + y],
                               pmz[size_t(k)*NZ + z]);
            dst[k] = solid[p] ? f0[i*Q + opp[k]] : f0[p*Q + k];
        }
    }
    f0.swap(f1);
}

static void lbm_step() { macro_collide(); stream(); }

// q_i = mean(u_i) over the whole cell — the Darcy superficial velocity.
static void volumeavg(double q[3])
{
    double su = 0, sv = 0, sw = 0;
    for(long i = 0; i < NN; i++) { su += U[i]; sv += V[i]; sw += W[i]; }
    q[0] = su/double(NN); q[1] = sv/double(NN); q[2] = sw/double(NN);
}

static void export_velocity(const string &path)
{
    ofstream out(path.c_str(), ios::binary);
    if(!out) die("cannot write " + path);
    int32_t hdr[4] = {NX, NY, NZ, 3};
    out.write(reinterpret_cast<const char*>(hdr), 16);
    vector<float> buf(3);
    for(long i = 0; i < NN; i++)                 // x fastest, as in the .raw
    {
        buf[0] = float(U[i]); buf[1] = float(V[i]); buf[2] = float(W[i]);
        out.write(reinterpret_cast<const char*>(buf.data()), 12);
    }
}

// --- one forcing direction -------------------------------------------------
static Solve solve(double fx, double fy, double fz, const Config &cfg, const char *tag)
{
    Solve out;
    reset_distributions();
    Fx = fx; Fy = fy; Fz = fz;

    long s = 0;
    int  hold = 0;
    bool have_prev = false;
    double pm[3] = {0,0,0}, pse[3] = {0,0,0};
    double bs[3] = {0,0,0}, bss[3] = {0,0,0};
    long   bn = 0;

    while(s < cfg.max_steps)
    {
        lbm_step(); s++;

        if(s % cfg.avg_sample == 0)
        {
            double q[3]; volumeavg(q);
            for(int c = 0; c < 3; c++)
            {
                if(!std::isfinite(q[c]))
                    die(string("non-finite flux in run ") + tag + " at step " + to_string(s));
                bs[c] += q[c]; bss[c] += q[c]*q[c];
            }
            bn++;
        }

        if(s % cfg.lag || bn < 2) continue;

        double m[3], se[3];
        for(int c = 0; c < 3; c++)
        {
            m[c]  = bs[c]/bn;
            double var = (bss[c] - bn*m[c]*m[c]) / (bn - 1);
            se[c] = sqrt(max(var, 0.0)/bn);
        }

        if(have_prev)
        {
            const double qmag = sqrt(m[0]*m[0] + m[1]*m[1] + m[2]*m[2]);
            const double flo  = cfg.qfloor * qmag;
            bool pass = true;
            for(int c = 0; c < 3; c++)
            {
                const double d   = fabs(m[c] - pm[c]);
                const double tol = max(cfg.eps * max(fabs(m[c]), flo),
                                       cfg.ksigma * sqrt(se[c]*se[c] + pse[c]*pse[c]));
                if(d >= tol) pass = false;
            }
            if(s >= cfg.min_steps && pass) hold++; else hold = 0;
        }

        if(!cfg.quiet)
            cout << "# " << tag << " " << s << "  q=(" << m[0] << ", " << m[1]
                 << ", " << m[2] << ")  hold=" << hold << endl;

        for(int c = 0; c < 3; c++) { pm[c] = m[c]; pse[c] = se[c]; bs[c] = bss[c] = 0.0; }
        bn = 0; have_prev = true;

        if(hold >= cfg.hold) break;
    }

    out.converged = (hold >= cfg.hold) ? 1 : 0;
    if(!out.converged && !cfg.quiet)
        cerr << "# " << tag << ": max-steps reached without convergence" << endl;

    // --- averaging phase ---------------------------------------------------
    double as[3] = {0,0,0}, ass[3] = {0,0,0};
    long   n = 0;
    for(long t = 1; t <= cfg.avg_steps; t++)
    {
        lbm_step(); s++;
        if(t % cfg.avg_sample) continue;
        double q[3]; volumeavg(q);
        for(int c = 0; c < 3; c++) { as[c] += q[c]; ass[c] += q[c]*q[c]; }
        n++;
    }

    if(n == 0) { volumeavg(out.q); }
    else
    {
        for(int c = 0; c < 3; c++)
        {
            out.q[c] = as[c]/n;
            if(n > 1)
            {
                double var = (ass[c] - n*out.q[c]*out.q[c]) / (n - 1);
                out.se[c] = sqrt(max(var, 0.0)/n);
            }
        }
    }
    out.steps = s;

    if(!cfg.velocity_prefix.empty())
        export_velocity(cfg.velocity_prefix + "." + tag + ".vel");

    return out;
}

// --- symmetric 3x3 eigen-decomposition (cyclic Jacobi) ---------------------
static void jacobi3(const double Ain[3][3], double val[3], double vec[3][3])
{
    double a[3][3];
    memcpy(a, Ain, sizeof(a));
    for(int i = 0; i < 3; i++) for(int j = 0; j < 3; j++) vec[i][j] = (i == j) ? 1.0 : 0.0;

    for(int sweep = 0; sweep < 50; sweep++)
    {
        double off = fabs(a[0][1]) + fabs(a[0][2]) + fabs(a[1][2]);
        if(off < 1e-18) break;
        for(int p = 0; p < 2; p++)
        for(int q = p+1; q < 3; q++)
        {
            if(fabs(a[p][q]) < 1e-300) continue;
            const double theta = (a[q][q] - a[p][p]) / (2.0*a[p][q]);
            const double t = (theta >= 0 ? 1.0 : -1.0) /
                             (fabs(theta) + sqrt(theta*theta + 1.0));
            const double c = 1.0/sqrt(t*t + 1.0), s = t*c;
            for(int k = 0; k < 3; k++)
            {
                const double akp = a[k][p], akq = a[k][q];
                a[k][p] = c*akp - s*akq;  a[k][q] = s*akp + c*akq;
            }
            for(int k = 0; k < 3; k++)
            {
                const double apk = a[p][k], aqk = a[q][k];
                a[p][k] = c*apk - s*aqk;  a[q][k] = s*apk + c*aqk;
                const double vkp = vec[k][p], vkq = vec[k][q];
                vec[k][p] = c*vkp - s*vkq; vec[k][q] = s*vkp + c*vkq;
            }
        }
    }
    for(int i = 0; i < 3; i++) val[i] = a[i][i];

    // sort descending, carrying the eigenvectors
    for(int i = 0; i < 2; i++)
    for(int j = i+1; j < 3; j++)
        if(val[j] > val[i])
        {
            swap(val[i], val[j]);
            for(int k = 0; k < 3; k++) swap(vec[k][i], vec[k][j]);
        }
}

// ---------------------------------------------------------------------------
static void usage()
{
    cerr <<
"Usage: lbm3d-perm <structure.raw> <results.csv> [options]\n"
"\n"
"Solves a triply periodic cell three times (force along +x, +y, +z) and\n"
"appends one row with the full 3x3 permeability tensor to <results.csv>.\n"
"\n"
"  --ff F            force factor; body force = 2.5e-07 * F   (default 1)\n"
"  --eps E           per-component relative tolerance         (default 1e-3)\n"
"  --lag N           steps between convergence checks         (default 2000)\n"
"  --hold K          consecutive passing checks required      (default 3)\n"
"  --min-steps N     no convergence before this step          (default 4000)\n"
"  --max-steps N     give up after this many steps            (default 200000)\n"
"  --avg-steps N     length of the averaging phase            (default 8000)\n"
"  --avg-sample N    flux sampling interval                   (default 25)\n"
"  --qfloor R        floor for the relative test              (default 1e-2)\n"
"  --ksigma K        accept a change within K sigma of the block mean's\n"
"                    own standard error                       (default 3)\n"
"  --velocity PREFIX write PREFIX.fx.vel, PREFIX.fy.vel, PREFIX.fz.vel\n"
"  --validate        extra solve along (1,1,1)/sqrt(3); reports the flux\n"
"                    predicted by K against the measured one (Koza09 test)\n"
"  --quiet           suppress per-check progress\n";
}

static long need_long(int argc, char **argv, int &i)
{
    if(i + 1 >= argc) { usage(); die(string("missing value for ") + argv[i]); }
    return strtol(argv[++i], nullptr, 10);
}
static double need_double(int argc, char **argv, int &i)
{
    if(i + 1 >= argc) { usage(); die(string("missing value for ") + argv[i]); }
    return strtod(argv[++i], nullptr);
}

int main(int argc, char **argv)
{
    if(argc < 3) { usage(); return 2; }
    build_opp();

    const string structure  = argv[1];
    const string resultfile = argv[2];
    Config cfg;

    for(int i = 3; i < argc; i++)
    {
        const string a = argv[i];
        if     (a == "--ff")         cfg.ff         = need_double(argc, argv, i);
        else if(a == "--eps")        cfg.eps        = need_double(argc, argv, i);
        else if(a == "--lag")        cfg.lag        = need_long(argc, argv, i);
        else if(a == "--hold")       cfg.hold       = int(need_long(argc, argv, i));
        else if(a == "--min-steps")  cfg.min_steps  = need_long(argc, argv, i);
        else if(a == "--max-steps")  cfg.max_steps  = need_long(argc, argv, i);
        else if(a == "--avg-steps")  cfg.avg_steps  = need_long(argc, argv, i);
        else if(a == "--avg-sample") cfg.avg_sample = need_long(argc, argv, i);
        else if(a == "--qfloor")     cfg.qfloor     = need_double(argc, argv, i);
        else if(a == "--ksigma")     cfg.ksigma     = need_double(argc, argv, i);
        else if(a == "--velocity")
        {
            if(i + 1 >= argc) { usage(); die("missing value for --velocity"); }
            cfg.velocity_prefix = argv[++i];
        }
        else if(a == "--validate")   cfg.validate = true;
        else if(a == "--quiet")      cfg.quiet    = true;
        else { usage(); die("unknown option " + a); }
    }

    read_structure(structure);
    build_neighbour_tables();
    allocate();

    const double porosity = double(n_pore)/double(NN);
    const double F = FX_BASE * cfg.ff;

    if(!cfg.quiet)
        cout << "# structure " << structure << "  " << NX << "x" << NY << "x" << NZ
             << "  porosity " << porosity << "  force " << F << "  mu " << MU << endl;

    const Solve rx = solve(F, 0, 0, cfg, "fx");
    const Solve ry = solve(0, F, 0, cfg, "fy");
    const Solve rz = solve(0, 0, F, cfg, "fz");

    // --- assemble K: column j is the response to a force along axis j -------
    double K[3][3];
    for(int c = 0; c < 3; c++)
    {
        K[c][0] = MU * rx.q[c] / F;
        K[c][1] = MU * ry.q[c] / F;
        K[c][2] = MU * rz.q[c] / F;
    }

    const double trace = K[0][0] + K[1][1] + K[2][2];
    const double k_xy = 0.5*(K[0][1] + K[1][0]);
    const double k_xz = 0.5*(K[0][2] + K[2][0]);
    const double k_yz = 0.5*(K[1][2] + K[2][1]);
    const double rec_xy = fabs(K[0][1] - K[1][0]) / fabs(trace);
    const double rec_xz = fabs(K[0][2] - K[2][0]) / fabs(trace);
    const double rec_yz = fabs(K[1][2] - K[2][1]) / fabs(trace);
    const double recip  = max(rec_xy, max(rec_xz, rec_yz));

    double Ks[3][3] = {{K[0][0], k_xy, k_xz},
                       {k_xy, K[1][1], k_yz},
                       {k_xz, k_yz, K[2][2]}};
    double val[3], vec[3][3];
    jacobi3(Ks, val, vec);

    const double kmean = trace/3.0;
    // Fractional anisotropy, the standard scale-free measure for a symmetric
    // positive tensor: 0 = isotropic, 1 = one non-zero principal value.
    const double dv = (val[0]-val[1])*(val[0]-val[1])
                    + (val[1]-val[2])*(val[1]-val[2])
                    + (val[2]-val[0])*(val[2]-val[0]);
    const double nrm = val[0]*val[0] + val[1]*val[1] + val[2]*val[2];
    const double fa  = nrm > 0 ? sqrt(0.5*dv/nrm) : NAN;

    if(cfg.validate)
    {
        const double c = 1.0/sqrt(3.0);
        const Solve rv = solve(F*c, F*c, F*c, cfg, "val");
        double pred[3], err = 0, qn = 0;
        for(int r = 0; r < 3; r++)
        {
            pred[r] = (Ks[r][0]*F*c + Ks[r][1]*F*c + Ks[r][2]*F*c) / MU;
            err += (rv.q[r]-pred[r])*(rv.q[r]-pred[r]);
            qn  += rv.q[r]*rv.q[r];
        }
        cout << setprecision(6);
        cout << "# tensor test along (1,1,1)  (converged " << rv.converged
             << ", " << rv.steps << " steps)" << endl;
        cout << "#   predicted q = (" << pred[0] << ", " << pred[1] << ", " << pred[2] << ")" << endl;
        cout << "#   measured  q = (" << rv.q[0] << ", " << rv.q[1] << ", " << rv.q[2] << ")" << endl;
        cout << "#   relative error " << sqrt(err)/(qn > 0 ? sqrt(qn) : 1) << endl;
    }

    // --- append one row -----------------------------------------------------
    bool need_header = true;
    { ifstream probe(resultfile.c_str(), ios::ate);
      if(probe.good() && probe.tellg() > 0) need_header = false; }

    ofstream csv(resultfile.c_str(), ios::app);
    if(!csv) die("cannot open " + resultfile + " for writing");

    if(need_header)
    {
        csv << "filename,ff,force,mu,nx,ny,nz,porosity,"
               "steps_fx,conv_fx,steps_fy,conv_fy,steps_fz,conv_fz,";
        const char *tg[3] = {"fx","fy","fz"};
        const char *cp[3] = {"x","y","z"};
        for(int r = 0; r < 3; r++) for(int c = 0; c < 3; c++)
            csv << "q" << cp[c] << "_" << tg[r] << ',';
        for(int r = 0; r < 3; r++) for(int c = 0; c < 3; c++)
            csv << "se_q" << cp[c] << "_" << tg[r] << ',';
        for(int r = 0; r < 3; r++) for(int c = 0; c < 3; c++)
            csv << "K_" << cp[r] << cp[c] << ',';
        csv << "k_xy,k_xz,k_yz,"
               "recip_xy,recip_xz,recip_yz,recip_resid,"
               "k1,k2,k3,k_mean,fa,"
               "v1_x,v1_y,v1_z,v2_x,v2_y,v2_z,v3_x,v3_y,v3_z\n";
    }

    csv << setprecision(10);
    csv << structure << ',' << cfg.ff << ',' << F << ',' << MU << ','
        << NX << ',' << NY << ',' << NZ << ',' << porosity << ','
        << rx.steps << ',' << rx.converged << ','
        << ry.steps << ',' << ry.converged << ','
        << rz.steps << ',' << rz.converged << ',';
    const Solve *runs[3] = {&rx, &ry, &rz};
    for(int r = 0; r < 3; r++) for(int c = 0; c < 3; c++) csv << runs[r]->q[c]  << ',';
    for(int r = 0; r < 3; r++) for(int c = 0; c < 3; c++) csv << runs[r]->se[c] << ',';
    for(int r = 0; r < 3; r++) for(int c = 0; c < 3; c++) csv << K[r][c] << ',';
    csv << k_xy << ',' << k_xz << ',' << k_yz << ','
        << rec_xy << ',' << rec_xz << ',' << rec_yz << ',' << recip << ','
        << val[0] << ',' << val[1] << ',' << val[2] << ',' << kmean << ',' << fa << ',';
    for(int j = 0; j < 3; j++) for(int i = 0; i < 3; i++)
        csv << vec[i][j] << (j == 2 && i == 2 ? '\n' : ',');
    csv.close();

    if(!cfg.quiet)
    {
        cout << setprecision(6);
        cout << "# K =" << endl;
        for(int r = 0; r < 3; r++)
            cout << "#   [ " << setw(12) << K[r][0] << " " << setw(12) << K[r][1]
                 << " " << setw(12) << K[r][2] << " ]" << endl;
        cout << "# reciprocity residuals  xy " << rec_xy << "  xz " << rec_xz
             << "  yz " << rec_yz << endl;
        cout << "# k1 " << val[0] << "  k2 " << val[1] << "  k3 " << val[2]
             << "  k_mean " << kmean << "  FA " << fa << endl;
    }

    return (rx.converged && ry.converged && rz.converged) ? 0 : 3;
}
