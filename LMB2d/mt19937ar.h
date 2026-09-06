#ifndef M_TW_H
#define M_TW_H

/* generates a random number on [0,1)-real-interval */
double genrand_real2(void);


void init_genrand(unsigned long s);
void init_by_array(unsigned long init_key[], int key_length);

#endif
