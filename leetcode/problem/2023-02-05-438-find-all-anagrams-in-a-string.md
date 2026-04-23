---
layout: leetcode-entry
title: "438. Find All Anagrams in a String"
permalink: "/leetcode/problem/2023-02-05-438-find-all-anagrams-in-a-string/"
leetcode_ui: true
entry_slug: "2023-02-05-438-find-all-anagrams-in-a-string"
---

[438. Find All Anagrams in a String](https://leetcode.com/problems/find-all-anagrams-in-a-string/description/) medium

[blog post](https://leetcode.com/problems/find-all-anagrams-in-a-string/solutions/3145307/kotlin-frequencies/)

```kotlin
    fun findAnagrams(s: String, p: String): List<Int> {
        val freq = IntArray(26) { 0 }
        var nonZeros = 0
        p.forEach {
            val ind = it.toInt() - 'a'.toInt()
            if (freq[ind] == 0) nonZeros++
            freq[ind]--
        }
        val res = mutableListOf<Int>()
        for (i in 0..s.lastIndex) {
            val currInd = s[i].toInt() - 'a'.toInt()
            if (freq[currInd] == 0) nonZeros++
            freq[currInd]++
            if (freq[currInd] == 0) nonZeros--
            if (i >= p.length) {
                val ind = s[i - p.length].toInt() - 'a'.toInt()
                if (freq[ind] == 0) nonZeros++
                freq[ind]--
                if (freq[ind] == 0) nonZeros--
            }
            if (nonZeros == 0) res += i - p.length + 1
        }
        return res
    }

```

#### Telegram
https://t.me/leetcode_daily_unstoppable/109
#### Intuition
We can count frequencies of `p` and then scan `s` to match them.

#### Approach
* To avoid checking a frequencies arrays, we can count how many frequencies are not matching, and add only when non-matching count is zero.
#### Complexity
- Time complexity:
  $$O(n)$$

- Space complexity:
  $$O(1)$$

