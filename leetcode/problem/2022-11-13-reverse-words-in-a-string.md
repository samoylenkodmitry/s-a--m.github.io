---
layout: leetcode-entry
title: "Reverse Words In A String"
permalink: "/leetcode/problem/2022-11-13-reverse-words-in-a-string/"
leetcode_ui: true
entry_slug: "2022-11-13-reverse-words-in-a-string"
---

[https://leetcode.com/problems/reverse-words-in-a-string/](https://leetcode.com/problems/reverse-words-in-a-string/) medium

A simple trick: reverse all the string, then reverse each word.

```kotlin

    fun reverseWords(s: String): String {
        val res = StringBuilder()
        val curr = Stack<Char>()
        (s.lastIndex downTo 0).forEach { i ->
            val c = s[i]
            if (c in '0'..'z') curr.push(c)
            else if (curr.isNotEmpty()) {
                if (res.length > 0) res.append(' ')
                while (curr.isNotEmpty()) res.append(curr.pop())
            }
        }
        if (curr.isNotEmpty() && res.length > 0) res.append(' ')
        while (curr.isNotEmpty()) res.append(curr.pop())
        return res.toString()
    }

```

Complexity: O(N)
Memory: O(N) - there is no O(1) solution for string in JVM

