---
layout: leetcode-entry
title: "440. K-th Smallest in Lexicographical Order"
permalink: "/leetcode/problem/2024-09-22-440-k-th-smallest-in-lexicographical-order/"
leetcode_ui: true
entry_slug: "2024-09-22-440-k-th-smallest-in-lexicographical-order"
---

[440. K-th Smallest in Lexicographical Order](https://leetcode.com/problems/k-th-smallest-in-lexicographical-order/description/) hard
[blog post](https://leetcode.com/problems/k-th-smallest-in-lexicographical-order/solutions/5819960/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/22092024-440-k-th-smallest-in-lexicographical?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/9k_BMYam9tU)
![1.webp](/assets/leetcode_daily_images/dd98f36f.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/743

#### Problem TLDR

`k` lexicographically smallest value from `1..n` #hard #math

#### Intuition

If we try the solution from the previous day https://t.me/leetcode_daily_unstoppable/742 it will give us TLE as the problem size is too big 10^9.
However, for Kotlin, the naive optimization of batch increments will pass:

```kotlin

val diff = min(nl, x + (10L - (x % 10L))) - x
if (i < k - diff) { x += diff; i += diff.toInt() }

```

The actual solution is to skip all numbers `x0..x9, x00..x99, x000..x999, x0000..x9999, xx00000..xx99999` for every prefix `x` while they are less than target `n`.

#### Approach

* steal someone else's solution

#### Complexity

- Time complexity:
$$O(lg(k) * lg(n))$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin

    fun findKthNumber(n: Int, k: Int): Int {
        var x = 1L; var i = 1; val nl = n.toLong()
        while (i < k) {
            if (x * 10L <= nl) x *= 10L else {
                if (x + 1L > nl) x /= 10L
                x++
                val diff = min(nl, x + (10L - (x % 10L))) - x
                if (i < k - diff) { x += diff; i += diff.toInt() }
                while (x > 0L && x % 10L == 0L) x /= 10L
            }
            i++
        }
        return x.toInt()
    }

```
```rust

    pub fn find_kth_number(n: i32, k: i32) -> i32 {
        let (mut x, mut i, n, k) = (1, 1, n as i64, k as i64);
        while i < k {
            let (mut count, mut from, mut to) = (0, x, x);
            while from <= n {
                count += to.min(n) - from + 1;
                from *= 10; to = to * 10 + 9
            }
            if i + count <= k { i += count; x += 1 }
            else { i += 1; x *= 10 }
        }
        x as i32
    }

```
```c++

    int findKthNumber(int n, int k) {
        long long x = 1, i = 1;
        while (i < k) {
            long long count = 0, from = x, to = x;
            while (from <= n) {
                count += min(to, static_cast<long long>(n)) - from + 1;
                from *= 10; to = to * 10 + 9;
            }
            if (i + count <= k) { i += count; x += 1; }
            else { i += 1; x *= 10; }
        }
        return static_cast<int>(x);
    }

```

