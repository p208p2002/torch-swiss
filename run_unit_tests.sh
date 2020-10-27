coverage run \
    --omit *dist-packages*,*site-packages*,unit_tests* \
    -m unittest unit_tests.test_metrics.TestMetrics

coverage report -m