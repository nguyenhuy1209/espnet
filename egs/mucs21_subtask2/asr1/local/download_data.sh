#!/bin/bash

#Copyright



# available_languages=(
#     "hi" "mr" "or" "ta" "te" "gu" "hi-en" "bn-en"
# )
available_languages=(
    "hi-en" "bn-en"
)
db=$1
lang=$2

if [ $# != 2 ]; then
    echo "Usage: $0 <db_root_dir> <spk>"
    echo "Available langauges for mucs subtask2: ${available_languages[*]}"
    exit 1
fi

if ! $(echo ${available_languages[*]} | grep -q ${lang}); then
    echo "Specified langauge (${lang}) is not available or not supported." >&2
    echo "Choose from: ${available_languages[*]}"
    exit 1
fi

declare -A trainset
# trainset['hi']='http://www.ee.iisc.ac.in/new/people/faculty/prasantg/downloads/Hindi_train.tar.gz'
# trainset['mr']='http://www.ee.iisc.ac.in/new/people/faculty/prasantg/downloads/Marathi_train.tar.gz'
# trainset['or']='http://www.ee.iisc.ac.in/new/people/faculty/prasantg/downloads/Odia_train.tar.gz'
# trainset['ta']='https://msropendata.com/datasets/7230b4b1-912d-400e-be58-f84e0512985e'
# trainset['te']='https://msropendata.com/datasets/7230b4b1-912d-400e-be58-f84e0512985e'
# trainset['gu']='https://msropendata.com/datasets/7230b4b1-912d-400e-be58-f84e0512985e'
trainset['hi-en']='http://www.ee.iisc.ac.in/new/people/faculty/prasantg/downloads/Hindi-English_train.tar.gz'
trainset['bn-en']='http://www.ee.iisc.ac.in/new/people/faculty/prasantg/downloads/Bengali-English_train.tar.gz'

declare -A testset
# testset['hi']='http://www.ee.iisc.ac.in/new/people/faculty/prasantg/downloads/Hindi_test.tar.gz'
# testset['mr']='http://www.ee.iisc.ac.in/new/people/faculty/prasantg/downloads/Marathi_test.tar.gz'
# testset['or']='http://www.ee.iisc.ac.in/new/people/faculty/prasantg/downloads/Odia_test.tar.gz'
# testset['ta']='https://msropendata.com/datasets/7230b4b1-912d-400e-be58-f84e0512985e'
# testset['te']='https://msropendata.com/datasets/7230b4b1-912d-400e-be58-f84e0512985e'
# testset['gu']='https://msropendata.com/datasets/7230b4b1-912d-400e-be58-f84e0512985e'
testset['hi-en']='http://www.ee.iisc.ac.in/new/people/faculty/prasantg/downloads/Hindi-English_test.tar.gz'
testset['bn-en']='http://www.ee.iisc.ac.in/new/people/faculty/prasantg/downloads/Bengali-English_test.tar.gz'

cwd=`pwd`
if [ ! -e ${db}/${lang}.done ]; then
    mkdir -p ${db}
    cd ${db}
    mkdir -p ${lang}
    cd ${lang}
    wget -O test.zip ${testset[$lang]}
    tar xf "test.zip"
    rm test.zip
    wget -O train.zip ${trainset[$lang]}
    tar xf "train.zip"
    rm train.zip
    cd $cwd
    echo "Successfully finished download."
    touch ${db}/${lang}.done
else
    echo "Already exists. Skip download."
fi
