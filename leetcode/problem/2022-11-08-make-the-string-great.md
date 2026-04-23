---
layout: leetcode-entry
title: "Make The String Great"
permalink: "/leetcode/problem/2022-11-08-make-the-string-great/"
leetcode_ui: true
entry_slug: "2022-11-08-make-the-string-great"
---

[https://leetcode.com/problems/make-the-string-great/](https://leetcode.com/problems/make-the-string-great/) easy

```kotlin

    fun makeGood(s: String): String {
        var ss = s.toCharArray()
        var finished = false
        while(!finished) {
            finished = true
            for (i in 0 until s.lastIndex) {
                if (ss[i] == '.') continue
                var j = i+1
                while(j <= s.lastIndex && ss[j] == '.') {
                    j++
                    continue
                }
                if (j == s.length) break

                var a = ss[i]
                var b = ss[j]
                if (a != b && Character.toLowerCase(a) ==
                        Character.toLowerCase(b)) {
                    ss[i] = '.'
                    ss[j] = '.'
                    finished = false
                }
            }
        }
        return ss.filter { it != '.' }.joinToString("")
    }

```

Explanation:
The simplest solution is just to simulate all the process, as input string is just 100 symbols.

Speed: O(n^2)
Memory: O(n)

