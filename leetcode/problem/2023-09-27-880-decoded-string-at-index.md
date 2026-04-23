---
layout: leetcode-entry
title: "880. Decoded String at Index"
permalink: "/leetcode/problem/2023-09-27-880-decoded-string-at-index/"
leetcode_ui: true
entry_slug: "2023-09-27-880-decoded-string-at-index"
---

[880. Decoded String at Index](https://leetcode.com/problems/decoded-string-at-index/description/) medium
[blog post](https://leetcode.com/problems/decoded-string-at-index/solutions/4095272/kotlin-you-know-the-length/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/27092023-880-decoded-string-at-index?r=2bam17&utm_campaign=post&utm_medium=web)
![image.png](/assets/leetcode_daily_images/d53f4c7d.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/352

#### Problem TLDR

`k`-th character in an encoded string like `a3b2=aaabaaab`

#### Intuition

We know the resulting length at every position of the encoded string. For example,
```
a3b2
1348
```
The next step, just walk from the end of the string and adjust `k`, by undoing repeating operation:
```
    // a2b2c2
    // 0 1 2 3 4 5 6 7 8 9 10 11 12 13
    // a a b a a b c a a b a  a  b  c
    // a2b2c2 = 2 x a2b2c = 2*(a2b2 + c) =
    // 2*(2*(a2 + b) + c) = 2*(2*(2*a + b) + c)
    //  k=9         9%(len(a2b2c)/2)
    //
    // a3b2    k=7
    // 12345678
    // aaabaaab
    // aaab    k=7%4=3
    //
    // abcd2    k=6
    // 12345678
    // abcdabcd  k%4=2
```

#### Approach

* use Long to avoid overflow
* check digit with `isDigit`
* Kotlin have a nice conversion function `digitToInt`
* corner case is when `search`` is become `0`, we must return first non-digit character

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin
    fun decodeAtIndex(s: String, k: Int): String {
      val lens = LongArray(s.length) { 1L }
      for (i in 1..s.lastIndex) lens[i] = if (s[i].isDigit())
          lens[i - 1] * s[i].digitToInt()
        else lens[i - 1] + 1
      var search = k.toLong()
      for (i in s.lastIndex downTo 0) if (s[i].isDigit())
          search = search % (lens[i] / s[i].digitToInt().toLong())
        else if (lens[i] == search || search == 0L) return "" + s[i]
      throw error("not found")
    }
```

