#!/usr/bin/env bash
git pull origin master
git tag -f latest
git push -f origin latest