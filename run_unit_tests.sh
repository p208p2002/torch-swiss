# python -m unittest unit_tests
coverage run \
    --source torch_swiss/ \
    -m unittest unit_tests

coverage report -m