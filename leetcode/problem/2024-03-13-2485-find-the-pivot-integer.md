---
layout: leetcode-entry
title: "2485. Find the Pivot Integer"
permalink: "/leetcode/problem/2024-03-13-2485-find-the-pivot-integer/"
leetcode_ui: true
entry_slug: "2024-03-13-2485-find-the-pivot-integer"
---

[2485. Find the Pivot Integer](https://leetcode.com/problems/find-the-pivot-integer/description/) easy
[blog post](https://leetcode.com/problems/find-the-pivot-integer/solutions/4867964/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/13032024-2485-find-the-pivot-integer?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/vhuJTxNMASg)
![2024-03-13_08-33.jpg](/assets/leetcode_daily_images/57595b31.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/537

#### Problem TLDR

Pivot of `1..n` where `sum[1..p] == sum[p..n]`. #easy

#### Intuition

Let's observe an example:
```j
  // 1 2 3 4 5 6 7 8
  // 1 2 3 4 5         5 * 6 / 2 = 15
  //           6 7 8   8 * 9 / 2 = 36 - 15
  //           p=6
  // p * (p + 1) / 2 == n * (n + 1) / 2 - p * (p - 1) / 2
```
The left part will increase with the grown of pivot `p`, so we can use Binary Search in that space.

Another solution is to simplify the equation more:
```j
  // x(x + 1)/2 == n(n + 1)/2 - x(x + 1)/2 + x
  // x(x + 1) - x == sum
  // x^2 == sum
```
Given that, just check if square root is perfect.

#### Approach

For more robust Binary Search:
* use inclusive `lo` and `hi`
* check the last condition `lo == hi`
* always move the boundaries: `lo = mi + 1`, `hi = mid - `
* use a separate condition to exit

#### Complexity

- Time complexity:
$$O(log(n))$$, square root is also log(n)

- Space complexity:
$$O(1)$$

#### Code

```kotlin

  fun pivotInteger(n: Int): Int {
    var lo = 1; var hi = n;
    while (lo <= hi) {
      val p = lo + (hi - lo) / 2
      val l = p * (p + 1) / 2
      val r = n * (n + 1) / 2 - p * (p - 1) / 2
      if (l < r) lo = p + 1 else
      if (l > r) hi = p - 1 else return p
    }
    return -1
  }

```
```rust

  pub fn pivot_integer(n: i32) -> i32 {
    let sum = n * (n + 1) / 2;
    let sq = (sum as f32).sqrt() as i32;
    if (sq * sq == sum) { sq } else { -1 }
  }

```

