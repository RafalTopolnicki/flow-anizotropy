// ---------------------------------------------------------------------------
// lbm2d-perm — permeability tensor of a 2D periodic porous cell
//
// Darcy's law for a doubly periodic cell is  q_i = (K_ij / mu) F_j, so the
// full 2x2 tensor needs two solves of the same structure: one driven by
// F = (F, 0), one by F = (0, F).  Each solve contributes one column of K:
//
//     run x:  k_xx = mu q_x / F ,  k_yx = mu q_y / F
//     run y:  k_xy = mu q_x / F ,  k_yy = mu q_y / F
//
// Both solves run in a single invocation so the structure is read once and
// the reciprocity residual |k_xy - k_yx| / (k_xx + k_yy) — which K = K^T
// makes an exact identity for Stokes flow — is available per sample as a
// free error bar on the off-diagonal.
//
// Convergence is judged on the two components of the domain-averaged flux
// separately, each against its own magnitude.  The transverse component is
// a heavily cancelling sum (its pointwise rms can be 50x its mean), so it
// is invisible both in a single-cross-section flux and in a whole-field L2
// residual, and it converges far later than the driven component.  A single
// passing check is not enough — the residuals fluctuate rather than decay
// once the transient is over — so --hold consecutive passes are required,
// and q is then block-averaged over a further --avg-steps rather than being
// read off at the stopping step.
//
// Normalisation note: q_i = sum(u_i) / (L * L0^2) reproduces the convention
// of the original volumeflux2d(), so k_xx here is directly comparable with
// the k column of the existing k0/k2/deeppore surveys.  Also unchanged: the
// reported velocity is the raw moment sum(f e)/rho, not the forced-LBM
// velocity sum(f e)/rho + F/(2 rho).  That omission biases the *diagonal*
// by ~F/2 / q (about 0.3% at ff=1) as a common factor and cancels from the
// off-diagonals, every ratio, and the anisotropy; it is left in place so
// that k stays comparable with the previously generated datasets.
//
//   Usage: lbm2d-perm <structure.gif|ppm|dat> <results.csv> [options]
// ---------------------------------------------------------------------------

#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <Magick++.h>

#include "lbm.h"

using namespace std;

// Legacy globals: the optional GL/particles translation unit declares these
// extern, so they stay defined here even though the headless build is the
// only one that matters for dataset generation.
float dt = 1;
int   mode = 1;
int   pause = 0;
float mnoznik_alpha = 0.1;
int   interponoff = 1;
int   visualization = 1;

static const double FX_BASE = 2.5e-07;   // body force at force factor 1

struct Config
{
    double ff          = 1.0;
    double eps         = 1e-3;
    long   lag         = 5000;
    int    hold        = 3;
    long   min_steps   = 10000;
    long   max_steps   = 500000;
    long   avg_steps   = 20000;
    long   avg_sample  = 50;
    double qfloor      = 1e-2;
    double ksigma      = 3.0;
    string velocity_prefix;
    bool   quiet       = false;
    double validate    = -1.0;   // if >= 0, extra solve at this force angle
};

struct Solve
{
    double qx = 0, qy = 0;          // block-averaged superficial flux
    double se_qx = 0, se_qy = 0;    // standard error of those means
    long   steps = 0;
    int    converged = 0;
    double fieldres = 0;            // diagnostic only
};

// ---------------------------------------------------------------------------

static void die(const string &msg)
{
    cerr << "lbm2d-perm: " << msg << endl;
    exit(2);
}

// One forcing direction, run to convergence and then block-averaged.
static Solve solve(double Fx, double Fy, const Config &cfg, const char *tag)
{
    Solve out;

    resetdistributions();
    fx = float(Fx);
    fy = float(Fy);

    // --- phase 1: run until the drift in BOTH flux components has stopped ---
    // Compare *block means*, not instantaneous samples.  Once the transient is
    // over the residuals stop decaying and fluctuate: on a converged structure
    // the instantaneous transverse flux still wobbles ~0.5% between checks, so
    // an instantaneous test never accumulates consecutive passes and runs to
    // the step cap on a solution that settled long ago.  Averaging within each
    // window turns the test back into a drift test.
    //
    // The tolerance is the looser of a relative bound and k-sigma of the block
    // mean's own standard error, so the run stops when the remaining drift is
    // statistically indistinguishable from the fluctuation.  That is
    // self-calibrating: no per-structure tuning of eps.
    long s = 0;
    int  hold = 0;
    bool have_prev = false;
    double pm_x = 0, pm_y = 0, pse_x = 0, pse_y = 0;
    double bsx = 0, bsy = 0, bsxx = 0, bsyy = 0;
    long   bn = 0;
    double last_mx = 0, last_my = 0;

    while(s < cfg.max_steps)
    {
        lbm();
        s++;

        if(s % cfg.avg_sample == 0)
        {
            double qx, qy;
            volumeavg(qx, qy);
            if(!std::isfinite(qx) || !std::isfinite(qy))
                die(string("non-finite flux in run ") + tag + " at step " + to_string(s));
            bsx += qx; bsxx += qx*qx;
            bsy += qy; bsyy += qy*qy;
            bn++;
        }

        if(s % cfg.lag || bn < 2) continue;

        const double mx = bsx / bn, my = bsy / bn;
        const double vx = (bsxx - bn*mx*mx) / (bn - 1);
        const double vy = (bsyy - bn*my*my) / (bn - 1);
        const double sex = sqrt(max(vx, 0.0) / bn);
        const double sey = sqrt(max(vy, 0.0) / bn);
        last_mx = mx; last_my = my;

        double dx = 0, dy = 0, tolx = 0, toly = 0;
        if(have_prev)
        {
            const double qmag = sqrt(mx*mx + my*my);
            const double flo  = cfg.qfloor * qmag;
            dx = fabs(mx - pm_x);
            dy = fabs(my - pm_y);
            tolx = max(cfg.eps * max(fabs(mx), flo),
                       cfg.ksigma * sqrt(sex*sex + pse_x*pse_x));
            toly = max(cfg.eps * max(fabs(my), flo),
                       cfg.ksigma * sqrt(sey*sey + pse_y*pse_y));

            if(s >= cfg.min_steps && dx < tolx && dy < toly) hold++;
            else                                             hold = 0;
        }

        if(!cfg.quiet)
            cout << "# " << tag << " " << s
                 << "  qx=" << mx << "  qy=" << my
                 << "  dx/tol=" << (tolx > 0 ? dx/tolx : 0)
                 << "  dy/tol=" << (toly > 0 ? dy/toly : 0)
                 << "  hold=" << hold << endl;

        pm_x = mx; pm_y = my; pse_x = sex; pse_y = sey; have_prev = true;
        bsx = bsy = bsxx = bsyy = 0.0; bn = 0;

        if(hold >= cfg.hold) break;
    }

    out.converged = (hold >= cfg.hold) ? 1 : 0;
    if(!out.converged && !cfg.quiet)
        cerr << "# " << tag << ": max-steps reached without convergence" << endl;

    // --- phase 2: average away the residual fluctuation --------------------
    // Past the transient the residuals stop decaying and fluctuate about the
    // steady state, so the instantaneous q at the stopping step is a noisy
    // estimate.  Averaging trailing samples cuts that by ~sqrt(n).
    snapshot_field();

    double sx = 0, sy = 0, sxx = 0, syy = 0;
    long   n = 0;

    for(long t = 1; t <= cfg.avg_steps; t++)
    {
        lbm();
        s++;
        if(t % cfg.avg_sample) continue;

        double qx, qy;
        volumeavg(qx, qy);
        if(!std::isfinite(qx) || !std::isfinite(qy))
            die(string("non-finite flux while averaging run ") + tag);

        sx += qx; sxx += qx*qx;
        sy += qy; syy += qy*qy;
        n++;
    }

    if(n == 0)
    {
        volumeavg(out.qx, out.qy);
        n = 1;
    }
    else
    {
        out.qx = sx / n;
        out.qy = sy / n;
        if(n > 1)
        {
            double vx = (sxx - n*out.qx*out.qx) / (n - 1);
            double vy = (syy - n*out.qy*out.qy) / (n - 1);
            out.se_qx = sqrt(max(vx, 0.0) / n);
            out.se_qy = sqrt(max(vy, 0.0) / n);
        }
    }

    out.steps    = s;
    out.fieldres = field_residual();

    if(!cfg.velocity_prefix.empty())
        exportvelocity(cfg.velocity_prefix + "." + tag + ".dat");

    return out;
}

// ---------------------------------------------------------------------------

static void usage()
{
    cerr <<
"Usage: lbm2d-perm <structure.gif|ppm|dat> <results.csv> [options]\n"
"\n"
"Solves the 256x256 doubly periodic cell twice (force along +x, then +y)\n"
"and appends one row with the full permeability tensor to <results.csv>.\n"
"\n"
"  --ff F            force factor; body force = 2.5e-07 * F   (default 1)\n"
"  --eps E           per-component relative tolerance         (default 1e-3)\n"
"  --lag N           steps between convergence checks         (default 5000)\n"
"  --hold K          consecutive passing checks required      (default 3)\n"
"  --min-steps N     no convergence before this step          (default 10000)\n"
"  --max-steps N     give up after this many steps            (default 500000)\n"
"  --avg-steps N     length of the averaging phase            (default 20000)\n"
"  --avg-sample N    flux sampling interval                   (default 50)\n"
"  --qfloor R        floor for the relative test, as a fraction of |q|\n"
"                                                             (default 1e-2)\n"
"  --ksigma K        also accept a change within K sigma of the block\n"
"                    mean's own standard error               (default 3)\n"
"  --velocity PREFIX write PREFIX.fx.dat and PREFIX.fy.dat\n"
"  --validate DEG    extra solve with the force at DEG degrees; reports the\n"
"                    flux predicted by K against the measured one (Koza09\n"
"                    tensor test).  Diagnostic — not written to the CSV.\n"
"  --quiet           suppress per-check progress on stdout\n";
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
    if(argc < 3) { usage(); return 1; }

    const string structure  = argv[1];
    const string resultfile = argv[2];

    Config cfg;
    for(int i = 3; i < argc; i++)
    {
        string a = argv[i];
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
        else if(a == "--validate")   cfg.validate   = need_double(argc, argv, i);
        else if(a == "--quiet")      cfg.quiet      = true;
        else if(a == "-h" || a == "--help") { usage(); return 0; }
        else { usage(); die("unknown option " + a); }
    }

    if(cfg.lag <= 0 || cfg.avg_sample <= 0) die("--lag and --avg-sample must be positive");

    Magick::InitializeMagick(*argv);

    const double F = FX_BASE * cfg.ff;

    initlbm(structure);
    const double porosity = getporosity_full();

    if(!cfg.quiet)
        cout << "# structure " << structure << "  porosity " << porosity
             << "  force " << F << " (ff=" << cfg.ff << ")  mu " << mu << endl;

    const Solve rx_ = solve(F, 0.0, cfg, "fx");
    const Solve ry_ = solve(0.0, F, cfg, "fy");

    // --- assemble the tensor ------------------------------------------------
    const double k_xx = mu * rx_.qx / F;
    const double k_yx = mu * rx_.qy / F;
    const double k_xy = mu * ry_.qx / F;
    const double k_yy = mu * ry_.qy / F;

    // K = K^T holds exactly for Stokes flow, so the residual is discretisation
    // error.  It scales with the trace, not with k_off, and therefore bounds
    // the off-diagonal only loosely — keep it per sample and use it to weight
    // or filter the off-diagonal target downstream.
    const double k_off  = 0.5 * (k_xy + k_yx);
    const double trace  = k_xx + k_yy;
    const double recip  = trace != 0.0 ? fabs(k_xy - k_yx) / fabs(trace) : NAN;

    const double km = 0.5 * (k_xx + k_yy);
    const double d1 = 0.5 * (k_xx - k_yy);
    const double d2 = k_off;
    const double r  = sqrt(d1*d1 + d2*d2);

    const double k1    = km + r;                       // major principal
    const double k2    = km - r;                       // minor principal
    const double aniso = km != 0.0 ? r / km : NAN;     // 0 = isotropic
    const double theta = 0.5 * atan2(d2, d1) * 180.0 / M_PI;   // (-90, 90]

    // Koza's alpha: signed rotation from the force direction to the flux,
    // counter-clockwise positive, so the two runs share one convention.
    const double alpha_x = atan2(rx_.qy, rx_.qx) * 180.0 / M_PI;
    const double alpha_y = atan2(ry_.qy, ry_.qx) * 180.0 / M_PI - 90.0;

    // --- optional tensor test ----------------------------------------------
    // K is fitted from two axis-aligned solves; if it is a genuine tensor,
    // a solve at any other force angle must reproduce q = (K/mu) F.  This is
    // the check in Koza09 section III.A and it validates the whole pipeline
    // (forcing, convergence, averaging, normalisation) in one number.
    if(cfg.validate >= 0.0)
    {
        const double th = cfg.validate * M_PI / 180.0;
        const double Fx = F * cos(th), Fy = F * sin(th);

        const Solve rv = solve(Fx, Fy, cfg, "val");

        // symmetrised K, since that is what would be used downstream
        const double pred_qx = (k_xx * Fx + k_off * Fy) / mu;
        const double pred_qy = (k_off * Fx + k_yy * Fy) / mu;
        const double qn      = sqrt(rv.qx*rv.qx + rv.qy*rv.qy);
        const double err     = sqrt((rv.qx-pred_qx)*(rv.qx-pred_qx)
                                  + (rv.qy-pred_qy)*(rv.qy-pred_qy)) / (qn > 0 ? qn : 1);

        cout << setprecision(6);
        cout << "# tensor test at " << cfg.validate << " deg  (converged "
             << rv.converged << ", " << rv.steps << " steps)" << endl;
        cout << "#   predicted q = (" << pred_qx << ", " << pred_qy << ")" << endl;
        cout << "#   measured  q = (" << rv.qx   << ", " << rv.qy   << ")" << endl;
        cout << "#   relative error " << err << endl;
    }

    // --- append one row -----------------------------------------------------
    bool need_header = true;
    {
        ifstream probe(resultfile.c_str(), ios::ate);
        if(probe.good() && probe.tellg() > 0) need_header = false;
    }

    ofstream csv(resultfile.c_str(), ios::app);
    if(!csv) die("cannot open " + resultfile + " for writing");

    if(need_header)
        csv << "filename,ff,force,mu,porosity,"
               "steps_fx,conv_fx,steps_fy,conv_fy,"
               "qx_fx,qy_fx,qx_fy,qy_fy,"
               "se_qx_fx,se_qy_fx,se_qx_fy,se_qy_fy,"
               "k_xx,k_yx,k_xy,k_yy,k_off,recip_resid,"
               "k1,k2,k_mean,anisotropy,theta_deg,alpha_x_deg,alpha_y_deg,"
               "fieldres_fx,fieldres_fy\n";

    csv << setprecision(10);
    csv << structure   << ',' << cfg.ff     << ',' << F          << ',' << mu << ',' << porosity << ','
        << rx_.steps   << ',' << rx_.converged << ',' << ry_.steps << ',' << ry_.converged << ','
        << rx_.qx      << ',' << rx_.qy     << ',' << ry_.qx     << ',' << ry_.qy     << ','
        << rx_.se_qx   << ',' << rx_.se_qy  << ',' << ry_.se_qx  << ',' << ry_.se_qy  << ','
        << k_xx        << ',' << k_yx       << ',' << k_xy       << ',' << k_yy       << ','
        << k_off       << ',' << recip      << ','
        << k1          << ',' << k2         << ',' << km         << ',' << aniso      << ','
        << theta       << ',' << alpha_x    << ',' << alpha_y    << ','
        << rx_.fieldres << ',' << ry_.fieldres << '\n';
    csv.close();

    if(!cfg.quiet)
    {
        cout << setprecision(6);
        cout << "# K = [ " << k_xx << " " << k_xy << " ; " << k_yx << " " << k_yy << " ]" << endl;
        cout << "# reciprocity residual " << recip
             << "   (relative error on k_off: "
             << (k_off != 0.0 ? fabs(k_xy - k_yx) / fabs(2*k_off) : NAN) << ")" << endl;
        cout << "# k1 " << k1 << "  k2 " << k2 << "  anisotropy " << aniso
             << "  theta " << theta << " deg" << endl;
    }

    // Non-convergence is a data-quality problem, not a crash: the row is
    // written either way and conv_fx / conv_fy flag it downstream.
    return (rx_.converged && ry_.converged) ? 0 : 3;
}
