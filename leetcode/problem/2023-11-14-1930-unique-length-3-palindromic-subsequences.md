---
layout: leetcode-entry
title: "1930. Unique Length-3 Palindromic Subsequences"
permalink: "/leetcode/problem/2023-11-14-1930-unique-length-3-palindromic-subsequences/"
leetcode_ui: true
entry_slug: "2023-11-14-1930-unique-length-3-palindromic-subsequences"
---

[1930. Unique Length-3 Palindromic Subsequences](https://leetcode.com/problems/unique-length-3-palindromic-subsequences/description/) medium
[blog post](https://leetcode.com/problems/unique-length-3-palindromic-subsequences/solutions/4285632/kotlin/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/14112023-1930-unique-length-3-palindromic?r=2bam17&utm_campaign=post&utm_medium=web)
![image.png](/assets/leetcode_daily_images/61f46b0c.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/403

#### Problem TLDR

Count of unique palindrome substrings of length 3

#### Intuition

We can count how many other characters between group of the current

#### Approach

Let's use Kotlin API:
* groupBy
* filterValues
* indexOf
* lastIndexOf

#### Complexity

- Time complexity:
$$O(n)$$, we can also use `withIndex` to avoid searching `indexOf` and `lastIndexOf`.

- Space complexity:
$$O(1)$$, if we store frequencies in an `IntArray`

#### Code

```kotlin

    fun countPalindromicSubsequence(s: String): Int {
      val freq = s.groupBy { it }.filterValues { it.size > 1 }
      var count = 0
      for ((l, f) in freq) {
        if (f.size > 2) count++
        val visited = HashSet<Char>()
        for (i in s.indexOf(l)..s.lastIndexOf(l))
          if (s[i] != l && visited.add(s[i])) count++
      }
      return count
    }

```

