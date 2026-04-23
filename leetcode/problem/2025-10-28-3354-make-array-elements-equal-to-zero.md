---
layout: leetcode-entry
title: "3354. Make Array Elements Equal to Zero"
permalink: "/leetcode/problem/2025-10-28-3354-make-array-elements-equal-to-zero/"
leetcode_ui: true
entry_slug: "2025-10-28-3354-make-array-elements-equal-to-zero"
---

[3354. Make Array Elements Equal to Zero](https://leetcode.com/problems/make-array-elements-equal-to-zero/description/) easy
[blog post](https://leetcode.com/problems/make-array-elements-equal-to-zero/solutions/7307251/kotlin-rust-by-samoylenkodmitry-vdh3/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/28102025-3354-make-array-elements?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/nwIJdXMazz8)

![20e665bd-064a-4623-bf91-44617549c8b7 (1).webp](/assets/leetcode_daily_images/14f61bdb.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1156

#### Problem TLDR

Places & directions that lead the simulation to all zeros #easy #simualtion

#### Intuition

Try every combination of place and direction

#### Approach

* without simulation: sum diff (left, right) must be less than 2

#### Complexity

- Time complexity:
$$O(n^3)$$, n^2 for the simulation

- Space complexity:
$$O(n)$$

#### Code

```kotlin
// 261ms
    fun countValidSelections(n: IntArray) =
        n.indices.sumOf { s -> if (n[s] == 0) listOf(-1,1).count { d ->
            val n = n.clone(); var c = s; var d = d
            while (c in 0..<n.size)
                if (n[c] > 0) { n[c]--; d *= -1; c += d } else c += d
            n.all { it == 0 }
        } else 0 }

```
```rust
// 2ms
    pub fn count_valid_selections(n: Vec<i32>) -> i32 {
        let (mut r, mut l, mut res) = (n.iter().sum::<i32>(),0,0);
        for n in n {
            l += n; r -= n;
            if n < 1 {
                if (l - r).abs() < 2 { res += 1 }
                if l == r { res += 1 }
            }
        } res
    }

```

