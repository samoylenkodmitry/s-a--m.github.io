---
layout: leetcode-entry
title: "2706. Buy Two Chocolates"
permalink: "/leetcode/problem/2023-12-20-2706-buy-two-chocolates/"
leetcode_ui: true
entry_slug: "2023-12-20-2706-buy-two-chocolates"
---

[2706. Buy Two Chocolates](https://leetcode.com/problems/buy-two-chocolates/description/) easy
[blog post](https://leetcode.com/problems/buy-two-chocolates/solutions/4428790/kotlin/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/20122023-2706-buy-two-chocolates?r=2bam17&utm_campaign=post&utm_medium=web&showWelcome=true)
![image.png](/assets/leetcode_daily_images/a726735a.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/445

#### Problem TLDR

Money change after two chocolates bought

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin

  fun buyChoco(prices: IntArray, money: Int): Int {
    var (a, b) = Int.MAX_VALUE to Int.MAX_VALUE
    for (x in prices)
      if (x < a) a = x.also { b = a }
      else if (x < b) b = x
    return (money - a - b).takeIf { it >= 0 } ?: money
  }

```

