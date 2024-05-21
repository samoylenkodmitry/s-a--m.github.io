---
layout: post
title: Setting up precommit lints for all team members
---

# What lints?

Current actively developed lints are Detekt and ktlint. klint have a `don't overengineer` philosophy while detekt is more configurable.

# How to set up Detekt?

[https://github.com/detekt/detekt](https://github.com/detekt/detekt)

Download binary from here [https://github.com/detekt/detekt/releases](https://github.com/detekt/detekt/releases)
Config file can be auto generated from command line or simple download mine [default-detekt-config.yml](https://gist.github.com/samoylenkodmitry/433572b16d22caa4a73d197ca92cbb69)

# How to set up ktlint?

[https://github.com/pinterest/ktlint](https://github.com/pinterest/ktlint)
Download binary from here [https://github.com/pinterest/ktlint/releases](https://github.com/pinterest/ktlint/releases)
Ktlint can be configured with standard `.editorconfig` file. You can find one anywhere in github or get mine: 
[.editorconfig](https://gist.github.com/samoylenkodmitry/5b7bc43160e042f716460c1d9ba784ee)

# How to make it check each commit?

Copy and configure script from here: 
[pre-commit](https://gist.github.com/samoylenkodmitry/0e988cd3445a0b390be20814eebce589)

```
#!/bin/bash
# Setting up guide http://dmitrysamoylenko.com/2020/12/06/configure_lints.html
# https://github.com/checkstyle/checkstyle
# https://github.com/pinterest/ktlint
# https://github.com/detekt/detekt

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

GIT_ROOT_DIR=$(git rev-parse --show-toplevel)

cd ${GIT_ROOT_DIR}
#echo ${GIT_ROOT_DIR}
f=()
while read line; do
  if [ -n "$line" ]; then
    if [[ "$line" =~ .*"/build/".* ]]; then
      true #skip generated code
    else
      f+=("$line")
    fi
  fi
done <<<"$(git diff --diff-filter=d --staged --name-only)"

filesJava=""
filesKt=""
countJava=0
countKt=0
for i in "${!f[@]}"; do
  if [[ "${f[i]}" == *.java ]]; then
    filesJava+=" ${f[i]}"
    countJava=$((countJava + 1))
  fi
  if [[ "${f[i]}" == *.kt ]]; then
    filesKt+=" ${f[i]}"
    countKt=$((countKt + 1))
  fi
done

for i in "${!f[@]}"; do
  if [[ "${f[i]}" =~ .*.(java|kt)$ ]]; then
    lineNum=0
    while IFS= read line; do
      if [ -n "$line" ]; then
        lineNum=$((lineNum+1))
        if [[ "$line" =~ ^\ +[^\*].* ]]; then
          echo "Line starts with spaces. Please apply project code style. File: ${f[i]}:$lineNum, line: $line"
          exit 1
        elif [[ "$line" =~ .*oleg.*|.*xoxoxo.* ]]; then
          echo "forbidden word in line. File: ${f[i]}:$lineNum, line: $line"
          exit 1
        else
          true #skip good code
        fi
      fi
    done < "${f[i]}"
  fi
done

if [ ${#filesJava} -eq 0 ]; then
  echo "No *.java files to check."
else
  configloc=-Dconfig_loc=${GIT_ROOT_DIR}/ivi/config/checkstyle
  config=${GIT_ROOT_DIR}/ivi/config/checkstyle/checkstyle.xml
  params="${configloc} -jar ${GIT_ROOT_DIR}/githooks/checkstyle-8.38-all.jar -c ${config}${filesJava}"

  ${JAVACMD} $params
  result=$?
  if [ $result -ne 0 ]; then
    echo "Please fix the checkstyle problems before submit the commit!"
    exit $result
  else
    echo "#java files: $countJava"
  fi
fi

if [ ${#filesKt} -eq 0 ]; then
  echo "No *.kt files to check."
  exit 0
fi

# ktlint check
git diff --diff-filter=d --staged --name-only | grep '\.kt[s"]\?$' | xargs ./ktlint .
result=$?
if [ $result -ne 0 ]; then
  echo "Please fix the ktlint problems before submit the commit!"
  exit $result
fi

#detekt check

check_by_detekt() {
  arg=$1
  files=${arg%?}
  if [ ${#files} -eq 0 ]; then
    true #skip
  else
    params="--fail-fast --config default-detekt-config.yml --input ${files}"
    ./detekt ${params}
    result=$?
    if [ $result -ne 0 ]; then
      echo "Please fix the detekt problems before submit the commit!"
      exit $result
    fi
  fi
}
count=0
filesd=""
for i in "${!f[@]}"; do
  if [[ "${f[i]}" == *.kt ]]; then
    filesd+="${f[i]},"
    count=$((count + 1))
    # split into batches
    if [ $count -gt 1000 ]; then
      check_by_detekt $filesd
      count=0
      filesd=""
    fi
  fi
done

check_by_detekt $filesd
echo "# kt files: $count"
exit 0

```

# How to make it work for all team members?

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
chmod +x ./ktlint
chmod +x ./detekt
chmod +x ./githooks/pre-commit
```
Now add the `githook/` directory and all new files to git and push it to server. All team should just open project again so AndroidStudio will run `build.gradle` script.
Notice that script support only unix os and should be specially edited to support windows.
