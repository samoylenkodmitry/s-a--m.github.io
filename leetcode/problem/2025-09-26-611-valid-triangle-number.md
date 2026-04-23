---
layout: leetcode-entry
title: "611. Valid Triangle Number"
permalink: "/leetcode/problem/2025-09-26-611-valid-triangle-number/"
leetcode_ui: true
entry_slug: "2025-09-26-611-valid-triangle-number"
---

[611. Valid Triangle Number](https://leetcode.com/problems/valid-triangle-number/description) medium
[blog post](https://leetcode.com/problems/valid-triangle-number/solutions/7225186/kotlin-rust-by-samoylenkodmitry-2wf4/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/26092025-611-valid-triangle-number?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/cYGMNH0eYUQ)

![1.webp](/assets/leetcode_daily_images/56a46be5.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1124

#### Problem TLDR

Count triangles can be made from numbers #medium #binary_search #two_sum

#### Intuition

Sort.
Binary Search: for every pair of numbers a and b search for c in [b-a..b+a]
Two Sum: same idea, but left border is always goes forward, so we can do increament instead of bs

#### Approach

* for the BinarySearch upper bound is already less than a+b, the range is [0..a]

#### Complexity

- Time complexity:
$$O(n^2log(n))$$ or n^2

- Space complexity:
$$O(1)$$

#### Code

```kotlin

// 113ms
    fun triangleNumber(n: IntArray): Int {
        n.sort()
        return (2..<n.size).sumOf { i ->
            (1..<i).sumOf { j ->
                var lo = 0; var hi = j-1
                while (lo <= hi) {
                    val m = (lo + hi) / 2
                    if (n[m] > n[i]-n[j]) hi = m - 1 else lo = m + 1
                }
                j - lo
            }
        }
    }

```
```kotlin

// 44ms
    fun triangleNumber(n: IntArray): Int {
        n.sort()
        return (2..<n.size).sumOf { i ->
            var l = 0; var r = i-1; var c = 0
            while (l < r) if (n[l] + n[r] > n[i]) c += r-- -l else ++l
            c
        }
    }

```

```rust

// 18ms
    pub fn triangle_number(mut n: Vec<i32>) -> i32 {
        n.sort_unstable();
        (2..n.len()).map(|i| {
            let (mut l, mut r, mut c) = (0, i-1, 0);
            while l < r { if n[l]+n[r] > n[i] { c += r-l; r-=1} else { l+=1}}
            c as i32
        }).sum()
    }

```

