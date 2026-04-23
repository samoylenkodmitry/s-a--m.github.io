---
layout: leetcode-entry
title: "1823. Find the Winner of the Circular Game"
permalink: "/leetcode/problem/2024-07-08-1823-find-the-winner-of-the-circular-game/"
leetcode_ui: true
entry_slug: "2024-07-08-1823-find-the-winner-of-the-circular-game"
---

[1823. Find the Winner of the Circular Game](https://leetcode.com/problems/find-the-winner-of-the-circular-game/description/) medium
[blog post](https://leetcode.com/problems/find-the-winner-of-the-circular-game/solutions/5438770/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/8072024-1823-find-the-winner-of-the?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/3otcyprlQEc)
![2024-07-08_08-27.webp](/assets/leetcode_daily_images/fb07c58b.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/663

#### Problem TLDR

Last of `k-th` excluded from `1..n` #medium #simulation #math

#### Intuition

Let's observe the problem:

```j

    // 1 2 3 4 5 1 2 3 4 5        2
    // * *
    //   x * *              2
    //       x * *          4
    //           x x * x *  1
    //                   x  5
    //
    // 1 2 3 4 5 1 3 5 3
    //   x   x   x   x

    // 6, 1
    // 1 2 3 4 5 6        1
    // x x x x x
```

I didn't see any simple pattern here (however, it exists, see below). So, let's just use a linked list and simulate the process.

The math solution involves knowing the Josephus Problem
https://en.wikipedia.org/wiki/Josephus_problem, and it is a Dynamic Programming `answer(n, k) = (answer(n - 1, k) + k) %n`, or `ans = 0; for (i in 1..n) ans = (ans + k) % i; ans + 1`.

#### Approach

Kotlin: let's implement linked list as an array of pointers.
Rust: let's implement a bottom up DP solution. (after reading the wiki and other's solutions :) )

#### Complexity

- Time complexity:
$$O(nk)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin

    fun findTheWinner(n: Int, k: Int): Int {
        val nexts = IntArray(n + 1) { it + 1 }; nexts[n] = 1
        var curr = 1
        repeat(n - 1) {
            var prev = curr
            repeat(k - 1) { prev = curr; curr = nexts[curr] }
            nexts[prev] = nexts[curr]
            curr = nexts[curr]
        }
        return curr
    }

```
```rust

    pub fn find_the_winner(n: i32, k: i32) -> i32 {
        let mut ans = 0;
        for i in 1..=n { ans = (ans + k) % i }
        ans + 1
    }

```

