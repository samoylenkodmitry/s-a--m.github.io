---
layout: post
title: Setting up precommit lints for all team members
---
# What lints?

Current actively developed lints are Detekt and ktlint. klint have a `don't overengener` philosophy while detekt is more configurable.

# How to set up Detekt?

https://github.com/detekt/detekt
Download binary from here https://github.com/detekt/detekt/releases
Config file can be auto generated from command line or simple download mine https://gist.github.com/samoylenkodmitry/433572b16d22caa4a73d197ca92cbb69

# How to set up ktlint?

https://github.com/pinterest/ktlint
Download binary from here https://github.com/pinterest/ktlint/releases
Ktlint can be configured with standard `.editorconfig` file. You can find one anywhere in github or get mine: 
https://gist.github.com/samoylenkodmitry/5b7bc43160e042f716460c1d9ba784ee

# How to make it check each commit?

Copy and configure script from here: 
https://gist.github.com/samoylenkodmitry/0e988cd3445a0b390be20814eebce589

```
#!/bin/bash
# https://github.com/checkstyle/checkstyle
# https://github.com/pinterest/ktlint
# https://github.com/detekt/detekt
# Edit the following paths to checkstyle, detekt and ktlint binaries and config files
GIT_ROOT_DIR=$(git rev-parse --show-toplevel)
configloc=-Dconfig_loc=${GIT_ROOT_DIR}/ivi/config/checkstyle
config=${GIT_ROOT_DIR}/ivi/config/checkstyle/checkstyle.xml
params="${configloc} -jar ${GIT_ROOT_DIR}/githooks/checkstyle-8.38-all.jar -c ${config}${files}"
ktlintbinary="./ktlint"
detektconfig="./default-detekt-config.yml"

# Determine the Java command to use to start the JVM.
if [ -n "$JAVA_HOME" ]; then
  if [ -x "$JAVA_HOME/jre/sh/java" ]; then
    # IBM's JDK on AIX uses strange locations for the executables
    JAVACMD="$JAVA_HOME/jre/sh/java"
  else
    JAVACMD="$JAVA_HOME/bin/java"
  fi
  if [ ! -x "$JAVACMD" ]; then
    die "ERROR: JAVA_HOME is set to an invalid directory: $JAVA_HOME

Please set the JAVA_HOME variable in your environment to match the
location of your Java installation."
  fi
else
  JAVACMD="java"
  which java >/dev/null 2>&1 || die "ERROR: JAVA_HOME is not set and no 'java' command could be found in your PATH.

Please set the JAVA_HOME variable in your environment to match the
location of your Java installation."
fi


cd ${GIT_ROOT_DIR}
#echo ${GIT_ROOT_DIR}
readarray -t f < <(git diff --diff-filter=d --staged --name-only)
files=""
for i in "${!f[@]}"; do
  files+=" ${f[i]}"
done

${JAVACMD} $params
result=$?
if [ $result -ne 0 ]; then
  echo "Please fix the checkstyle problems before submit the commit!"
  exit $result
fi

#ktlint check
git diff --diff-filter=d --staged --name-only | grep '\.kt[s"]\?$' | xargs ${ktlintbinary} .
result=$?
if [ $result -ne 0 ]; then
  echo "Please fix the ktlint problems before submit the commit!"
  exit $result
fi

#detekt check
filesd=""
for i in "${!f[@]}"; do
  filesd+="${f[i]},"
done
filesd=${filesd%?}
params="--fail-fast --config ${detektconfig} --input ${filesd}"
./detekt ${params}
result=$?
if [ $result -ne 0 ]; then
  echo "Please fix the detekt problems before submit the commit!"
  exit $result
fi
exit 0

```

# How to make it run for all team members?

Make folder `githooks/` in the root git project directory and put all downloaded files into it.
Then in top of the project `build.gradle` insert:

```
exec {
	executable './../enable_lints.sh'
}
```

This will execute the script `enable_lints.sh` that will apply git path to the `githooks/` directory and make git execute the script `pre-commit` before each commit. Contents of the script:

enable_lints.sh
```
#!/bin/bash
git config --global core.hooksPath githooks
```

Remember to make each script executable
```
chmod +x filename
```
