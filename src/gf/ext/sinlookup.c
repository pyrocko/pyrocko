/**
 * Example for a sine/cosine table lookup
 *
 * (main file that uses sin1.h/sin1.c)
 *
 * @file sinlookup.c
 * @author stfwi
 *
 */
#include "lut.h"
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/types.h>
 
/**
 * Generate the lookup tables for sin1() in a way that the text can be pasted
 * into a C source code.
 *
 * @return void
 */
void generate_tables()
{
  #define Q15 (1.0/(double)((1<<15)-1))
  #define TABLE_SIZE  (1<<5)
  #define SCALER ((M_PI/2.0) / TABLE_SIZE)
  int i;
  printf("static int16_t sin90_offset[TABLE_SIZE+1] = {\n  ");
  for(i=0; i < TABLE_SIZE; i++) {
    printf("0x%04x%s", (int16_t) (sin(SCALER * i) / Q15), (i%8!=7) ? "," : ",\n  ");
  }
  printf("0x7fff\n};\n");
}
 
/**
 * Run a test comparing the real floating point sine with the sin1 function.
 * Output CSV: "ANGLE in DEG", "REAL SINE", "SIN1", "ERROR".
 *
 * @return void
 */
void run()
{
  double angle;
  printf("%6s, %8s, %8s, %6s\n", "angle", "sin", "sin1", "error", "cos", "cos1", "error");
  for(angle=0; angle<360; angle+=0.1) {
    double lookup_sine = sin1(angle * 32768.0 / 360.0) * Q15;
    double real_sine   = sin(angle * 2*M_PI / 360.0);
    double sine_error  = real_sine - lookup_sine;

    double lookup_cos = cos1(angle * 32768.0 / 360.0) * Q15;
    double real_cos   = cos(angle * 2*M_PI / 360.0);
    double cos_error  = real_cos - lookup_cos;
    printf("%6.1f, %+8.5f, %+8.5f, %+8.6f, %+8.5f, %+8.5f, %+8.6f\n", angle, real_sine, lookup_sine, sine_error, real_cos, lookup_cos, cos_error);
  }
}
 
/**
 * Test main function
 *
 * @param int argc
 * @param char **argv
 * @return int
 */
int main(int argc, char** argv)
{
  if(argc < 2) {
    fprintf(stderr, "Usage %s -g          : Create lookup tables\n", argv[0]);
    fprintf(stderr, "      %s <int angle> : Test a number (0 to 32767)\n", argv[0]);
    fprintf(stderr, "      %s -r          : Run iterated numbers test\n", argv[0]);
    return 1;
  } else if(!strcmp(argv[1], "-g")) {
    generate_tables();
  } else if(!strcmp(argv[1], "-r")) {
    run();
  } else if(!isnan(atof(argv[1]))) {
    long l = (long) atof(argv[1]);
    while(l <  0x0000) l += 0x8000;
    while(l >= 0x7fff) l -= 0x8000;
    printf("%d\n", (int) sin1(l));
  }
  return 0;
}
