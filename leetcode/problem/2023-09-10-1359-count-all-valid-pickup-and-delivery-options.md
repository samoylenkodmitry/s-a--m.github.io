---
layout: leetcode-entry
title: "1359. Count All Valid Pickup and Delivery Options"
permalink: "/leetcode/problem/2023-09-10-1359-count-all-valid-pickup-and-delivery-options/"
leetcode_ui: true
entry_slug: "2023-09-10-1359-count-all-valid-pickup-and-delivery-options"
---

[1359. Count All Valid Pickup and Delivery Options](https://leetcode.com/problems/count-all-valid-pickup-and-delivery-options/description/) hard
[blog post](https://leetcode.com/problems/count-all-valid-pickup-and-delivery-options/solutions/4024574/kotlin-the-pattern/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/10092023-1359-count-all-valid-pickup?utm_campaign=post&utm_medium=web)
![image.png](/assets/leetcode_daily_images/48517602.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/335

#### Problem TLDR

Count permutations of the `n` `pickup -> delivery` orders

#### Intuition

Let's look at how orders can be placed and draw the picture:
```bash
      // 1: p1 d1            variantsCount = 1
      // 2:                  length = 2
      // "___p1____d1_____": vacantPlaces = 3
      //              p2 d2
      //        p2       d2
      // p2              d2
      //        p2 d2
      // p2        d2
      // p2 d2
      //                                variantsCount = 6
      // 3:                             length = 4
      // "___p1____d1____p2____d2____": vacantPlaces = 5
      //                         p3 d3
      //                    p3      d3
      //              p3            d3
      //        p3                  d3
      // p3                         d3
      //                    p3 d3
      //              p3       d3             x6
      //        p3             d3
      // p3                    d3
      //              p3 d3
      //        p3       d3
      // p3              d3
      //        p3 d3
      // p3        d3
      // p3 d3
```
In this example, we can see the pattern:
* the number of vacant places grows by `2` each round
* inside each round there are repeating parts of arithmetic sum, that can be reused

#### Approach

* use `Long` to avoid overflow

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin

    fun countOrders(n: Int): Int {
      var variantsCount = 1L
      var currSum = 1L
      var item = 1L
      val m = 1_000_000_007L
      repeat(n - 1) {
        item = (item + 1L) % m
        currSum = (currSum + item) % m
        item = (item + 1L) % m
        currSum = (currSum + item) % m
        variantsCount = (variantsCount * currSum) % m
      }
      return variantsCount.toInt()
    }

```

