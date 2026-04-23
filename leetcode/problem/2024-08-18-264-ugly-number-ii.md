---
layout: leetcode-entry
title: "264. Ugly Number II"
permalink: "/leetcode/problem/2024-08-18-264-ugly-number-ii/"
leetcode_ui: true
entry_slug: "2024-08-18-264-ugly-number-ii"
---

[264. Ugly Number II](https://leetcode.com/problems/ugly-number-ii/description/) medium
[blog post](https://leetcode.com/problems/ugly-number-ii/solutions/5653905/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/18082024-264-ugly-number-ii?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/Xd5bzjQlp1U)
![1.webp](/assets/leetcode_daily_images/ee071348.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/706

#### Problem TLDR

`n`th number with only [1,2,3,5] multipliers #medium #heap

#### Intuition

First, understand the problem: the number should be divided only by the 1, 2, 3, and 5 dividers.
The simple way is to maintain a sorted set of numbers and peek the lowest from it.

There is a clever solution exists, however: maintain separate pointers for 2, 3 and 5. The sorted set is a sequence of all the results for [1..n] and each pointer must point to the lowest not yet multiplied value.
There is a corner case, when pointer of `3` points to number `2`, and vice versa, pointer of `2` points to the `3`. To handle the duplicate result of `2 x 3 = 6 and 3 x 2 = 6`, compare each pointer that is equal to the current result.

#### Approach

* for the first approach, we can use PriorityQueue with the HashSet, or just TreeSet
* the number can overflow the 32-bit value

#### Complexity

- Time complexity:
$$O(nlog(n))$$ for the TreeSet, $$O(n)$$ for the clever

- Space complexity:
$$O(n)$$ for the TreeSet, $$O(1)$$ for the clever

#### Code

```kotlin

    fun nthUglyNumber(n: Int) = TreeSet<Long>().run {
        add(1)
        repeat(n - 1) {
            val curr = pollFirst()
            add(curr * 2); add(curr * 3); add(curr * 5)
        }
        first().toInt()
    }

```
```rust

    pub fn nth_ugly_number(n: i32) -> i32 {
        let (mut u, m, mut p) =
            (vec![1; n as usize], [2, 3, 5], [0, 0, 0]);
        for i in 1..u.len() {
            u[i] = p.iter().zip(m)
                .map(|(&p, m)| u[p] * m).min().unwrap();
            for (p, m) in p.iter_mut().zip(m) {
                if u[*p] * m == u[i] { *p += 1 }}
        }
        u[u.len() - 1]
    }

```

