---
layout: leetcode-entry
title: "1436. Destination City"
permalink: "/leetcode/problem/2023-12-15-1436-destination-city/"
leetcode_ui: true
entry_slug: "2023-12-15-1436-destination-city"
---

[1436. Destination City](https://leetcode.com/problems/destination-city/description/) easy
[blog post](https://leetcode.com/problems/destination-city/solutions/4406829/kotlin/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/15122023-1436-destination-city?r=2bam17&utm_campaign=post&utm_medium=web&showWelcome=true)
![image.png](/assets/leetcode_daily_images/e93723c9.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/439

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$, with `toSet`

#### Code

```kotlin

    fun destCity(paths: List<List<String>>): String =
      (paths.map { it[1] } - paths.map { it[0] }).first()

```

