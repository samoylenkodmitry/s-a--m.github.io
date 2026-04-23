---
layout: leetcode-entry
title: "Reverse Vowels Of A String"
permalink: "/leetcode/problem/2022-11-04-reverse-vowels-of-a-string/"
leetcode_ui: true
entry_slug: "2022-11-04-reverse-vowels-of-a-string"
---

[https://leetcode.com/problems/reverse-vowels-of-a-string/](https://leetcode.com/problems/reverse-vowels-of-a-string/) easy

Solution [kotlin]

```kotlin

    fun reverseVowels(s: String): String {
        val vowels = setOf('a', 'e', 'i', 'o', 'u', 'A', 'E', 'I', 'O', 'U')
        var chrs = s.toCharArray()
        var l = 0
        var r = chrs.lastIndex
        while(l < r) {
            while(l<r && chrs[l] !in vowels) l++
            while(l<r && chrs[r] !in vowels) r--
            if (l < r) chrs[l] = chrs[r].also { chrs[r] = chrs[l] }
            r--
            l++
        }
        return String(chrs)
    }

```

Explanation:
Straightforward solution : use two pointers method and scan from the both sides.

Speed: O(N), Memory O(N)

