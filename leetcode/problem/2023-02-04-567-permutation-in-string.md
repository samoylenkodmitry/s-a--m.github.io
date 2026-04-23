---
layout: leetcode-entry
title: "567. Permutation in String"
permalink: "/leetcode/problem/2023-02-04-567-permutation-in-string/"
leetcode_ui: true
entry_slug: "2023-02-04-567-permutation-in-string"
---

[567. Permutation in String](https://leetcode.com/problems/permutation-in-string/description/) medium

[blog post](https://leetcode.com/problems/permutation-in-string/solutions/3139851/kotlin-frequencies/?orderBy=most_votes)

```kotlin
    fun checkInclusion(s1: String, s2: String): Boolean {
        val freq1 = IntArray(26) { 0 }
        s1.forEach {  freq1[it.toInt() - 'a'.toInt()]++  }
        val freq2 = IntArray(26) { 0 }
        for (i in 0..s2.lastIndex) {
            freq2[s2[i].toInt() - 'a'.toInt()]++
            if (i >= s1.length) freq2[s2[i - s1.length].toInt() - 'a'.toInt()]--
            if (Arrays.equals(freq1, freq2)) return true
        }
        return false
    }

```

#### Telegram
https://t.me/leetcode_daily_unstoppable/108
#### Intuition
We can count the chars frequencies in the `s1` string and use the sliding window technique to count and compare char frequencies in the `s2`.
#### Approach
* to decrease cost of comparing arrays, we can also use hashing
#### Complexity
- Time complexity:
  $$O(n)$$
- Space complexity:
  $$O(1)$$

