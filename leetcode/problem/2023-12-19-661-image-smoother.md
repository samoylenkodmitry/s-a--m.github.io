---
layout: leetcode-entry
title: "661. Image Smoother"
permalink: "/leetcode/problem/2023-12-19-661-image-smoother/"
leetcode_ui: true
entry_slug: "2023-12-19-661-image-smoother"
---

[661. Image Smoother](https://leetcode.com/problems/image-smoother/description/) easy
[blog post](https://leetcode.com/problems/image-smoother/solutions/4424198/kotlin/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/19122023-661-image-smoother?r=2bam17&utm_campaign=post&utm_medium=web&showWelcome=true)
![image.png](/assets/leetcode_daily_images/374c7469.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/444

#### Problem TLDR

3x3 average of each cell in 2D matrix

#### Complexity

- Time complexity:
$$O(nm)$$

- Space complexity:
$$O(nm)$$

#### Code

```kotlin

  fun imageSmoother(img: Array<IntArray>): Array<IntArray> =
    Array(img.size) {
      val ys = (max(0, it - 1)..min(img.lastIndex, it + 1)).asSequence()
      IntArray(img[0].size) {
        val xs = (max(0, it - 1)..min(img[0].lastIndex, it + 1)).asSequence()
        ys.flatMap { y -> xs.map { img[y][it] } }.average().toInt()
      }
    }

```

