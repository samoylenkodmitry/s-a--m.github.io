---
layout: leetcode-entry
title: "2938. Separate Black and White Balls"
permalink: "/leetcode/problem/2024-10-15-2938-separate-black-and-white-balls/"
leetcode_ui: true
entry_slug: "2024-10-15-2938-separate-black-and-white-balls"
---

[2938. Separate Black and White Balls](https://leetcode.com/problems/separate-black-and-white-balls/description/) medium
[blog post](https://leetcode.com/problems/separate-black-and-white-balls/solutions/5914908/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/15102024-2938-separate-black-and?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/ptl7zYHaPx0)
![1.webp](/assets/leetcode_daily_images/4472c228.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/769

#### Problem TLDR

Min moves to sort `01` string #medium #greedy

#### Intuition

Let's try to do this for each of `1` in our example:

```j

    // 0123456789
    // 1001001001
    //       .**  2
    //    .****   4
    // .******    6 = 12

```
There is a pattern: the number of moves to push each `1` to the right is equal to the number of `0` between it and its final position. So, going from the end and counting zeros is the answer.

#### Approach

* we can make iteration forward and count `1` instead to speed up and shorten the code
* some arithmetic is also applicable (to remove `if` branching)

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin

    fun minimumSteps(s: String): Long {
        var x = 0L
        return s.sumOf { x += it - '0'; x * ('1' - it) }
    }

```
```rust

    pub fn minimum_steps(s: String) -> i64 {
        let mut x = 0;
        s.bytes().map(|b| {
            if b > b'0' { x += 1; 0 } else { x }
        }).sum()
    }

```
```c++

    long long minimumSteps(string s) {
        long long x = 0, res = 0;
        for (auto c: s) res += ('1' - c) * (x += c - '0');
        return res;
    }

```

