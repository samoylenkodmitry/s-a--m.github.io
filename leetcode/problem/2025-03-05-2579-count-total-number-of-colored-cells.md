---
layout: leetcode-entry
title: "2579. Count Total Number of Colored Cells"
permalink: "/leetcode/problem/2025-03-05-2579-count-total-number-of-colored-cells/"
leetcode_ui: true
entry_slug: "2025-03-05-2579-count-total-number-of-colored-cells"
---

[2579. Count Total Number of Colored Cells](https://leetcode.com/problems/count-total-number-of-colored-cells/description/) medium
[blog post](https://leetcode.com/problems/count-total-number-of-colored-cells/solutions/6498715/kotlin-rust-by-samoylenkodmitry-93u4/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/05032025-2579-count-total-number?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/kAoXb1fq4-w)
![1.webp](/assets/leetcode_daily_images/ea50551c.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/915

#### Problem TLDR

Arithmetic sum #medium #math

#### Intuition
![x.webp](/assets/leetcode_daily_images/5fc80b8e.webp)
The diagonal wall grows one item at a time.

```j

    // 1
    // 1 + 4 = 5
    // 5 + 8 = 13
    // 13 + 12 = f(n - 1) + n * 2 + (n - 2) * 2

```

Arithmetic sum of n:
```j
coloredCells(n) = coloredCells(1) + ∑(i=2 to n) (i*4 - 4)
                = 1 + 4*∑(i=2 to n) i - 4*(n-1)
                = 1 + 4*[n(n+1)/2 - 1] - 4*(n-1)
                = 1 + 4*[n(n+1)/2 - 1] - 4n + 4
                = 1 + 2n(n+1) - 4 - 4n + 4
                = 1 + 2n² + 2n - 4 - 4n + 4
                = 2n² - 2n + 1
```

#### Approach

* draw, notice the pattern, write the code
* ask claude for the math formula

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$ for the recursion

#### Code

```kotlin

    fun coloredCells(n: Int): Long =
        if (n < 2) 1 else coloredCells(n - 1) + n * 4 - 4

```
```rust

    pub fn colored_cells(n: i32) -> i64 {
        let n = n as i64; 2 * n * n - 2 * n + 1
    }

```
```c++

    long long coloredCells(int n) {
       long long x = n; return 2 * x * x - 2 * x + 1;
    }

```

