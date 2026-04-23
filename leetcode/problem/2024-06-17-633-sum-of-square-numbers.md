---
layout: leetcode-entry
title: "633. Sum of Square Numbers"
permalink: "/leetcode/problem/2024-06-17-633-sum-of-square-numbers/"
leetcode_ui: true
entry_slug: "2024-06-17-633-sum-of-square-numbers"
---

[633. Sum of Square Numbers](https://leetcode.com/problems/sum-of-square-numbers/description/) medium
[blog post](https://leetcode.com/problems/sum-of-square-numbers/solutions/5324625/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/17062024-633-sum-of-square-numbers?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/8lKbxRUWyyQ)
![2024-06-17_05-53.webp](/assets/leetcode_daily_images/da7cca5d.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/642

#### Problem TLDR

Is `c` sum of squares? #medium #binary_search

#### Intuition

From simple brute force of `0..c` for `a` and `b` we can do the following optimizations:
* use `sqrt` upper bound O(n^2) -> O((sqrt(n))^2)
* notice that `sum` function grows linearly and we can do a Binary Search of `c` in it O((sqrt(n))^2) -> O(sqrt(n)log(n))
* the trickiest part: `a` and `b` can themselves be the upper and lower bounds -> O(sqrt(n))

#### Approach

Let's implement both solutions.

#### Complexity

- Time complexity:
$$O(sqrt(n)log(n))$$ and $$O(sqrt(n))$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin

    fun judgeSquareSum(c: Int): Boolean {
        val s = Math.sqrt(c.toDouble()).toLong()
        for (a in 0..s) {
            var lo = 0L; var hi = s
            while (lo <= hi) {
                val mid = lo + (hi - lo) / 2
                val sum = a * a + mid * mid
                if (sum == c.toLong()) return true
                if (sum > c.toLong()) hi = mid - 1
                else lo = mid + 1
            }
        }
        return false
    }

```
```rust

    pub fn judge_square_sum(c: i32) -> bool {
        let (mut lo, mut hi) = (0u64, (c as f64).sqrt() as u64);
        while lo <= hi {
            let sum = lo * lo + hi * hi;
            if sum == c as u64 { return true }
            if sum > c as u64 { hi -= 1 } else { lo += 1 }
        }; false
    }

```

