#include <stdio.h>
#include <stdlib.h>

#include <check.h>

#include <libswiftnav/linear_algebra.h>
#include <libswiftnav/plover/check_plover.h>

#include "check_utils.h"

START_TEST(test_plover)
{
  fail_unless(check_plover_main() == 0);
}
END_TEST

Suite* plover_test_suite(void)
{
  Suite *s = suite_create("Plover tests");

  TCase *tc_core = tcase_create("Core");
  tcase_add_test(tc_core, test_plover);
  suite_add_tcase(s, tc_core);
  return s;
}
