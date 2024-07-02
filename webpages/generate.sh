#!/bin/sh

jupyter nbconvert --to notebook --execute ./webpages/public_procurements.ipynb --output public_procurements.ipynb

# Convert the notebook to html
quarto render ./webpages/public_procurements.ipynb

mv /webpages/public_procurements.html /webpages/www/public_procurements.html
mv /webpages/public_procurements_files /webpages/www


