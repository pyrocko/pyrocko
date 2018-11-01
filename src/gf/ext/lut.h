/**
 * Example for a interpolated sine/cosine table lookup
 *
 * @file sin1.h
 * @author stfwi
 *
 */
#ifndef __SW__SIN1_H__
#define __SW__SIN1_H__
#ifdef  __cplusplus
extern "C" {
#endif
 
#include <sys/types.h>
 
/**
 * Sine calculation using interpolated table lookup.
 * Instead of radiants or degrees we use "turns" here. Means this
 * sine does NOT return one phase for 0 to 2*PI, but for 0 to 1.
 * Input: -1 to 1 as int16 Q15  == -32768 to 32767.
 * Output: -1 to 1 as int16 Q15 == -32768 to 32767.
 *
 * @param int16_t angle Q15
 * @return int16_t Q15
 */
int16_t sin1(int16_t angle);
 
/**
 * Cosine calculation using interpolated table lookup.
 * Instead of radiants or degrees we use "turns" here. Means this
 * cosine does NOT return one phase for 0 to 2*PI, but for 0 to 1.
 * Input: -1 to 1 as int16 Q15  == -32768 to 32767.
 * Output: -1 to 1 as int16 Q15 == -32768 to 32767.
 *
 * @param int16_t angle Q15
 * @return int16_t Q15
 */
int16_t cos1(int16_t angle);

int16_t tan1(int16_t angle);
 
#ifdef  __cplusplus
}
#endif
#endif
