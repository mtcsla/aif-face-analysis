#!/bin/bash
set -e

if ! command -v poetry &> /dev/null
then
    echo "Poetry could not be found. Please install poetry first."
    exit 1
fi

poetry run pip3 freeze > requirements.txt

echo "requirements.txt has been updated with frozen dependencies from poetry."
