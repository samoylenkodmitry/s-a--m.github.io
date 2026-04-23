---
layout: leetcode-entry
title: "168. Excel Sheet Column Title"
permalink: "/leetcode/problem/2023-08-22-168-excel-sheet-column-title/"
leetcode_ui: true
entry_slug: "2023-08-22-168-excel-sheet-column-title"
---

[168. Excel Sheet Column Title](https://leetcode.com/problems/excel-sheet-column-title/description/) easy
[blog post](https://leetcode.com/problems/excel-sheet-column-title/solutions/3943534/kotlin-math/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/22082023-168-excel-sheet-column-title?utm_campaign=post&utm_medium=web)

![image.png](/assets/leetcode_daily_images/3839bd9d.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/316

#### Problem TLDR

Excel col number to letter-number `1` -> `A`, `28` -> `AB`

#### Intuition

Just arithmetic conversion of number to string with radix of `26` instead of `10`. Remainder from division by 26 gives the last letter. Then the number must be divided by 26.

#### Approach
* use a StringBuilder
* number must be `n-1`

#### Complexity

- Time complexity:
$$O(log(n))$$, logarithm by radix of 26

- Space complexity:
$$O(log(n))$$

#### Code

```kotlin

    fun convertToTitle(columnNumber: Int): String = buildString {
      var n = columnNumber
      while (n > 0) {
        insert(0, ((n - 1) % 26 + 'A'.toInt()).toChar())
        n = (n - 1) / 26
      }
    }

```

