---
layout: leetcode-entry
title: "443. String Compression"
permalink: "/leetcode/problem/2023-03-02-443-string-compression/"
leetcode_ui: true
entry_slug: "2023-03-02-443-string-compression"
---

[443. String Compression](https://leetcode.com/problems/string-compression/description/) medium

[blog post](https://leetcode.com/problems/string-compression/solutions/3246608/kotlin-contradiction-in-the-description/)

```kotlin

fun compress(chars: CharArray): Int {
    var end = 0
    var curr = 0
    while (curr < chars.size) {
        val c = chars[curr++]
        var currCount = 1
        while (curr < chars.size && c == chars[curr]) {
            curr++
            currCount++
        }
        chars[end++] = c
        if (currCount > 1) currCount.toString().forEach { chars[end++] = it }
    }
    return end
}

```

#### Join me on telegram
https://t.me/leetcode_daily_unstoppable/135
#### Intuition
You don't need to split a number into groups of `9`'s.
The right way to convert number `123` into a string is to divide it by 10 each time, then reverse a part of the array.

#### Approach
* Let's just do a naive `toString` for simplicity.
* to avoid mistakes with indexes, use explicit variable for count the duplicate chars
#### Complexity
- Time complexity:
$$O(n)$$
- Space complexity:
$$O(lg_10(n))$$, for storing `toString`. For this task it is a `4`

