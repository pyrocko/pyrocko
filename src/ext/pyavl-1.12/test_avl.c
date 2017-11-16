
 /* File "test_avl.c" */

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <time.h>
#include <string.h>
#include "avl.h"

/* TEST */

static int
item_comp(void *unused, const void *k1, const void *k2)
{
  int n1 = *((int *) k1), n2 = *((int *) k2);

  (void) unused;
  return n1 < n2 ? -1 : n1 != n2 ? +1 : 0;
}

static void
item_print(const void *k, void *p)
{

  int *lnpos = p != NULL ? (int *) p : NULL;

  fprintf(stdout, "%4d", *((int *) k));
  if (p && ++*lnpos == 20)
	{
	  *lnpos = 0;
	  fprintf(stdout, "%c", '\n');
	}
}

#define NUM_ELEMENTS(ar)	(sizeof(ar) / sizeof(ar[0]))

#define HEADER(str)	printf("[Test %s]\n",str)
#define TAIL(str)	printf("[Test %s done]\n",str)

#if 0==1
static void
print_arr(int ar[], int n)
{
  int *p;

  for (p = &ar[0]; --n >= 0;)
	{
	  fprintf(stdout, "%4d", *p++);
	}
  fprintf(stdout, "%c", '\n');
}
#endif

static void
init_arr(int ar[], int n, int lo)
{
  int j, *p;

  for (p = &ar[0], j = lo; --n >= 0; *p++ = j++)
	;
}

/* 
 * C FAQ by Steve Summit
 * <http://www.eskimo.com/~scs/C-faq/>
 * 
 * Question 13.16
 * 
 * How can I get random integers in a certain range?
 * 
 * The obvious way,
 * 
 *     rand() % N        == POOR ==
 * 
 * (which tries to return numbers from 0 to N-1) is poor, because the
 * low-order bits of many random number generators are distressingly
 * non-random.  (See question 13.18.)  A better method is something like
 * 
 *     (int)((double)rand() / ((double)RAND_MAX + 1) * N)
 * 
 * If you're worried about using floating point, you could use
 * 
 *     rand() / (RAND_MAX / N + 1)
 * 
 * Both methods obviously require knowing RAND_MAX (which ANSI #defines in
 * <stdlib.h>), and assume that N is much less than RAND_MAX.
 */

static void
shuffle_arr(int ar[], int n)
{
  int j, p;

  while (--n > 0)
	{
	  j = rand() / (RAND_MAX / (n+1) + 1);
	  p = ar[j];
	  ar[j] = ar[n];
	  ar[n] = p;
	}
}

/* [t] is an existing tree handle */
static void
load_arr(avl_tree t, int ar[], int n, avl_bool_t verify)
{
  int *p;

  for (p = &ar[0]; n; n--, p++)
	{
	  /* say true to allow duplicates */
	  (void) avl_ins(p, t, avl_false);
	  if (verify)
		assert(avl_verify(t));
	}
}

void
ins_proc(avl_tree t, int ar[], int n, avl_bool_t shuffle)
{

  int *r, *p;

  HEADER("ins");
  avl_empty(t);
  if (shuffle)
	shuffle_arr(ar, n);
  load_arr(t, ar, n, /*verify */ avl_true);
  for (p = &ar[0]; n; n--, p++)
	{
	  assert((r = avl_find(p, t)) != NULL && *r == *p);
	}
  TAIL("ins");
  fflush(stdout);
}

#if 0==0
void
del_proc(avl_tree t, int *ar, int n, avl_bool_t shuffle)
{

  avl_tree w = NULL;
  int *p, *car;
  void *b[1];

  HEADER("del (this will take longer than testing ins)");
  if (avl_isempty(t))
	load_arr(t, ar, n, avl_true);
  car = malloc(n * sizeof(int));

  if (car == NULL)
	return;
  memcpy(car, ar, n * sizeof(int));

  if (shuffle)
	shuffle_arr(car, n);
  for (p = &car[0]; n; n--, p++)
	{
	  w = avl_dup(t, NULL);
	  assert(avl_verify(w));
	  (void) avl_del(p, w, b);
	  assert(avl_verify(w));
	  assert(*((int *) *b) == *p);
	  /* Include if duplicates not allowed */
#if 0==0
	  assert(avl_find(p, w) == NULL);
#endif
	  avl_destroy(w);
	}
  free(car);
  TAIL("del");
  fflush(stdout);
}
#endif

/* ar: array of indices */
void
del_index_proc(avl_tree t, avl_bool_t shuffle)
{

  avl_tree w = NULL;
  int n, *p, *s, *ar;
  void *b[1];

  HEADER("del by index");
  if ((n = avl_size(t)) == 0)
	return;
  ar = malloc(n * sizeof(int));

  if (ar == NULL)
	{
	  HEADER("interrupted, sorry !\n");
	  return;
	}
  init_arr(ar, n, 1);
  if (shuffle)
	shuffle_arr(ar, n);
  for (p = &ar[0]; n; n--, p++)
	{
	  w = avl_dup(t, NULL);
	  assert(avl_verify(w));
	  s = avl_find_index(*p, t);
	  (void) avl_del_index(*p, w, b);
	  assert(*s == *((int *) *b));
	  assert(avl_verify(w));
	  /* Include if duplicates not allowed */
#if 0==1
	  assert(avl_find(p, w) == NULL);
#endif
	  avl_destroy(w);
	}
  free(ar);
  TAIL("del");
  fflush(stdout);
}

#define TEST_DELMIN avl_true
#define TEST_DELMAX avl_false

#if 0==0
void
delextr_proc(avl_tree t, avl_bool_t testmin)
{

  avl_code_t(*del_func) (avl_tree, void **);
  int j;

  HEADER(testmin ? "delmin" : "delmax");
  assert(avl_verify(t));
  if (avl_isempty(t))
	{
	  fprintf(stdout, "\t%s\n", "delextr_proc: empty handle");
	  fflush(stdout);
	  return;
	}

  {
	avl_tree w = avl_dup(t, NULL);

	assert(avl_verify(w));
	del_func = testmin == avl_true ? avl_del_first : avl_del_last;
	for (j = avl_size(w); j; j--)
	  {
		(void) del_func(w, NULL);
		assert(avl_verify(w));
	  }
	assert(avl_isempty(w));
	avl_destroy(w);
  }

  TAIL(testmin ? "delmin" : "delmax");
  fflush(stdout);
}
#endif
#if 0==0
void
cat_proc(int *ar0, int n0, int *ar1, int n1)
{

  avl_tree t0, t1;

  HEADER("cat");
  t0 =
	avl_create(item_comp, avl_default_item_copy, avl_default_item_dispose,
			   malloc, free, NULL);
  t1 =
	avl_create(item_comp, avl_default_item_copy, avl_default_item_dispose,
			   malloc, free, NULL);
  load_arr(t0, ar0, n0, avl_false);
  assert(avl_verify(t0));
  load_arr(t1, ar1, n1, avl_false);
  assert(avl_verify(t1));
  avl_cat(t0, t1);
  assert(avl_size(t0) == n0 + n1);

  {
	int j, *p, *r;

	for (p = &ar0[0], j = n0; j; j--, p++)
	  assert((r = avl_find(p, t0)) != NULL && *r == *p);
	for (p = &ar1[0], j = n1; j; j--, p++)
	  assert((r = avl_find(p, t0)) != NULL && *r == *p);
  }
  avl_destroy(t0);
  avl_destroy(t1);
  TAIL("cat");
  fflush(stdout);
}
#endif

#if 0==0
void
split_proc(avl_tree t, int *ar, int n, avl_bool_t shuffle)
{

  avl_tree t0, t1;
  int *car;

  HEADER("split (please wait)");
  if (avl_isempty(t))
	load_arr(t, ar, n, avl_false);
  car = malloc(n * sizeof(int));

  if (car == NULL)
	return;
  memcpy(car, ar, n * sizeof(int));

  if (shuffle)
	shuffle_arr(car, n);
  t0 =
	avl_create(item_comp, avl_default_item_copy, avl_default_item_dispose,
			   malloc, free, NULL);
  t1 =
	avl_create(item_comp, avl_default_item_copy, avl_default_item_dispose,
			   malloc, free, NULL);

  {
	avl_tree w;
	int *p, N, cnt;

	N = avl_size(t);
	for (p = car, cnt = 0; n; n--, p++)
	  {
		w = avl_dup(t, NULL);
		assert(avl_verify(w));
		avl_split(p, w, t0, t1);
		assert(avl_verify(t0));
		assert(avl_verify(t1));
		assert(avl_size(t0) + avl_size(t1) == N - 1);
		avl_empty(t0);
		avl_empty(t1);
		free(w);
		w = NULL;
		if (++cnt == 80)
		  {
			fprintf(stdout, "%c", '.');
			cnt = 0;
			fflush(stdout);
		  }
	  }
  }

  free(car);
  fprintf(stdout, "%c", '\n');
  TAIL("split");
  fflush(stdout);
}
#endif

#if 0==0
void
iterator_proc(avl_tree t, avl_bool_t display)
{

  avl_iterator cur;
  const void *p;
  int n, col;

  HEADER("iterator");
  fprintf(stdout, "\tsize of traversed tree = %d\n", avl_size(t));
  cur = avl_iterator_new(t, AVL_ITERATOR_INI_PRE);
  for (n = 0, col = 0; (p = avl_iterator_next(cur)) != NULL; n++)
	{
	  if (display == avl_true)
		item_print(p, &col);
	}
  if (display)
	printf("%c", '\n');
  assert(avl_iterator_next(cur) == NULL);
  assert(n == avl_size(t));
  while (avl_iterator_prev(cur) != NULL)
	--n;
  assert(n == 0);
  /* second part */
  avl_iterator_kill(cur);
  cur = avl_iterator_new(t, AVL_ITERATOR_INI_POST);
  for (n = 0, col = 0; (p = avl_iterator_prev(cur)) != NULL; n++)
	{
	  if (display == avl_true)
		item_print(p, &col);
	}
  if (display)
	printf("%c", '\n');
  assert(avl_iterator_prev(cur) == NULL);
  assert(n == avl_size(t));
  while (avl_iterator_next(cur) != NULL)
	--n;
  assert(n == 0);
  avl_iterator_kill(cur);
  TAIL("iterator");
  fflush(stdout);
}
#endif

int
main(void)
{

  avl_tree t;
  int *ar, N;

  srand(time((time_t *) NULL));

  t =
	avl_create(item_comp, avl_default_item_copy, avl_default_item_dispose,
			   malloc, free, NULL);

  /* size of tree for tests */
  N = 10000;
  ar = malloc(N * sizeof(int));

  if (ar == NULL)
	return 1;
  init_arr(ar, N, 0);

#if 0==0
  ins_proc(t, ar, N, avl_true);
  assert(avl_size(t) == N);
  if (N < 40)
	{
	  avl_walk(t, item_print, NULL);
	  fprintf(stdout, "%s", "\n\n");
	}
#endif

#if 0==1
  {
	print_arr(ar, N);
	shuffle_arr(ar, N);
	print_arr(ar, N);
  }
#endif

#if 0==0
  {
	avl_tree w = avl_dup(t, NULL);

	assert(avl_verify(w));
	avl_destroy(w);
  }
#endif

  iterator_proc(t, N < 200 ? avl_true : avl_false);
  /*exit(0); */

#if 0==0
  del_proc(t, ar, N, avl_true);
  /*avl_walk(t,item_print,NULL); */

  del_index_proc(t, avl_true);

  delextr_proc(t, TEST_DELMIN);
  delextr_proc(t, TEST_DELMAX);
#endif
#if 0==0
  {
	int ar[3000], *ar0, *ar1, n0, n1;

	n0 = 2000;
	n1 = 1000;
	init_arr(ar0 = &ar[0], n0, 1);
	init_arr(ar1 = ar0 + n0, n1, n0);
	shuffle_arr(ar0, n0);
	shuffle_arr(ar1, n1);
	cat_proc(ar0, n0, ar1, n1);
  }
#endif

#if 0==0
  {
	printf("\tsize of split tree = %d\n", avl_size(t));
	split_proc(t, ar, N, avl_true);
  }
#endif

  avl_destroy(t);
  free(ar);
  fflush(stdout);

  return 0;
}
