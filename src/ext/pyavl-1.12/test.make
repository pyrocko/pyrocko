# -*-make-*-
OBJ_DIR		= build/test.osx
SRC			= avl.c test_avl.c
OBJ			= $(SRC:%.c=$(OBJ_DIR)/%.o)

CFLAGS		= -c -O2 -Wall -Wno-parentheses -Wno-uninitialized
DEFS		= -DHAVE_AVL_VERIFY -DAVL_SHOW_ERROR_ON

.PHONY: test
test: test_avl
	./test_avl

test_avl: $(OBJ)
	cc $(OBJ) -o $@

$(OBJ): $(OBJ_DIR)/%.o: %.c
	[ -d $(OBJ_DIR) ] || mkdir -p $(OBJ_DIR)
	cc $(CFLAGS) $(DEFS) $< -o $@

clean: 
	rm -f $(OBJ)
	rm -f test_avl
	rmdir $(OBJ_DIR)
