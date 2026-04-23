---
layout: leetcode-entry
title: "781. Rabbits in Forest"
permalink: "/leetcode/problem/2025-04-20-781-rabbits-in-forest/"
leetcode_ui: true
entry_slug: "2025-04-20-781-rabbits-in-forest"
---

[781. Rabbits in Forest](https://leetcode.com/problems/rabbits-in-forest/description) medium
[blog post](https://leetcode.com/problems/rabbits-in-forest/solutions/6669530/kotlin-rust-by-samoylenkodmitry-epo4/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/20042025-781-rabbits-in-forest?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/a2Bwo9E0RDo)
![1.webp](/assets/leetcode_daily_images/d26a1610.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/964

#### Problem TLDR

Count total rabbits from other numbers #medium #math #brainteaser

#### Intuition

* the rabbit tells how many *other* rabbits are

Thoughts process:

```j

    // 10,10,10     11   how?

    // brain teaser
    // 10 red
    // 10 red
    // 10 red   it is 3 that answered,
    //          10 - 3 = 7 not answered

    // 2 2 2 2 2
    // 5 +
    // I do not understand the problem
    // "have the same color as you"
    // is it including or excluding?
    // suppose excluding
    // 1 1 2
    // *     one other - mark red
    //   *   one other -  mark red
    //     * two others - no other with two
    // 3 + 2 of no-matches

    // 10 10 10
    // *         ten others - mark red (total 1 + 10)
    //    *      ten others - red
    //       *   ten others - red
    // total = 11

    // 2 2 2
    // *     2 others (total 1 + 2 = 3)
    //   *   2 others
    //     * 2 others
    // 3 of red

    // 1 1 = 2
    // 1 1 1 = 1 1 | 1 = 2 + 2
    // 1 1 1 1 = 1 1 | 1 1 = 2 + 2
    // 1 1 1 1 1 = 1 1 | 1 1 | 1  = 2 + 2 + 2 = 6

    // 2 2 2 2
    // so we have 4 rabbits
    // only 3 can be same color (2 + 1)
    // x = 2
    // group = 3 = (2 + 1) = g = x + 1
    // buckets = f(2) / (2 + 1) = f / g + f % g
    //         = 4 / 3 + 4 % 3 = 2
    // count = buckets * g = 2 * 3 = 6

    // 2 = 3
    // 2 2 = 3
    // 2 2 2 = 3
    // 2 2 2 2 = 2 2 2 | 2 = 3 + 3 = 6

```

* each group defined by the `others` answer, `group count` is `others + 1`
* total answered rabbits should be split into the buckes of `group count`

#### Approach

* from `u/votrubac/` & `u/lee215/`: we can increment a new group on the go

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin

    fun numRabbits(answers: IntArray) = answers.groupBy { it }
    .entries.sumOf { (x, f) -> (f.size + x) / (x + 1) * (x + 1) }

```
```kotlin

    fun numRabbits(answers: IntArray): Int {
        val f = IntArray(1001); for (x in answers) ++f[x]
        for (x in 0..999) if (f[x] > 0)
            f[1000] += (f[x] + x) / (x + 1) * (x + 1)
        return f[1000]
    }

```
```kotlin

    fun numRabbits(answers: IntArray): Int {
        val f = IntArray(1000); var r = 0;
        for (x in answers) if (f[x]++ % (x + 1) < 1) r += x + 1
        return r
    }

```
```rust

    pub fn num_rabbits(answers: Vec<i32>) -> i32 {
        let (mut f, mut r) = ([0; 1000], 0);
        for x in answers {
            if f[x as usize] % (x + 1) < 1 { r += x + 1 }
            f[x as usize] += 1
        } r
    }

```
```c++

    int numRabbits(vector<int>& a) {
       int r = 0, f[1000];
       for (int x: a) if (f[x]++ % (x + 1) < 1) r += x + 1;
       return r;
    }

```

