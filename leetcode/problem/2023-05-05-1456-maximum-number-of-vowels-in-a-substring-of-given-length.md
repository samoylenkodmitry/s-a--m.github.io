---
layout: leetcode-entry
title: "1456. Maximum Number of Vowels in a Substring of Given Length"
permalink: "/leetcode/problem/2023-05-05-1456-maximum-number-of-vowels-in-a-substring-of-given-length/"
leetcode_ui: true
entry_slug: "2023-05-05-1456-maximum-number-of-vowels-in-a-substring-of-given-length"
---

[1456. Maximum Number of Vowels in a Substring of Given Length](https://leetcode.com/problems/maximum-number-of-vowels-in-a-substring-of-given-length/description/) medium

```kotlin

fun maxVowels(s: String, k: Int): Int {
    val vowels = setOf('a', 'e', 'i', 'o', 'u')
    var count = 0
    var max = 0
    for (i in 0..s.lastIndex) {
        if (s[i] in vowels) count++
        if (i >= k && s[i - k] in vowels) count--
        if (count > max) max = count
    }
    return max
}

```

[blog post](https://leetcode.com/problems/maximum-number-of-vowels-in-a-substring-of-given-length/solutions/3487078/kotlin-sliding-window/)
[substack](https://dmitriisamoilenko.substack.com/p/leetcode-daily-5052023?sd=pf)
#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/203
#### Intuition
Count vowels, increasing them on the right border and decreasing on the left of the sliding window.
#### Approach
* we can use `Set` to check if it is a vowel
* look at `a[i - k]` to detect if we must start move left border from `i == k`
#### Complexity
- Time complexity:
$$O(n)$$
- Space complexity:
$$O(1)$$

