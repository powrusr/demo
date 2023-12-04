#!/bin/bash
limit=$(wc -l < pdf_urls.txt)
counter=0
for PDF in $(cat pdf_urls.txt)
do
  echo "$PDF"
  ((counter=counter+1))
  curl --parallel-max 3 --location "$PDF" --output "$counter".pdf
done
