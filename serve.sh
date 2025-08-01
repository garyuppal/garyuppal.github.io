#!/bin/bash

set -e  # exit on first error

echo "📦 Configuring bundler..."
bundle config set --local path 'vendor/bundle'

echo "🔧 Installing dependencies..."
bundle install

echo "🚀 Starting Jekyll server..."
bundle exec jekyll serve
