#!/usr/bin/env python
# -*- coding:utf-8 -*-

from setuptools import setup, find_packages

setup(
    name="ood_estimators",
    version='0.1',
    description="Estimators of out-of-distribution performance.",
    long_description="Utilities for an elimination of in-domain overfitting and generability evaluation.",
    classifiers=[],
    author="To Be Added",
    author_email="tobeadded@tobeadded.com",
    url="gitlab.com",
    license="MIT",
    packages=find_packages(include=["ood_estimators"]),
    use_scm_version={"write_to": ".version", "write_to_template": "{version}\n"},
    setup_requires=["setuptools_scm"],
    include_package_data=True,
    zip_safe=True,
    install_requires=[
        "torch>=1.7",
        "transformers==4.5.1",
        "sentencepiece==0.1.95",
        "scikit-learn==0.24.1",
        "sacrebleu==1.5.1",
        "spacy==2.3.5"
    ],
    # package_data={"reflexive_diaries": ["annotations/*", "models/configs/*"]},
)
