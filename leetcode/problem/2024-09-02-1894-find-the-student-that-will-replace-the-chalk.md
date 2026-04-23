---
layout: leetcode-entry
title: "1894. Find the Student that Will Replace the Chalk"
permalink: "/leetcode/problem/2024-09-02-1894-find-the-student-that-will-replace-the-chalk/"
leetcode_ui: true
entry_slug: "2024-09-02-1894-find-the-student-that-will-replace-the-chalk"
---

[1894. Find the Student that Will Replace the Chalk](https://leetcode.com/problems/find-the-student-that-will-replace-the-chalk/description/) medium
[blog post](https://leetcode.com/problems/find-the-student-that-will-replace-the-chalk/solutions/5724001/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/02092024-1894-find-the-student-that?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/_8MT4fyjo-w)
![1.webp](/assets/leetcode_daily_images/325df625.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/722

#### Problem TLDR

Position of a `k` sum in a cyclic array #medium

#### Intuition

First, eliminate the full loops, then find the position.
To find it, we can just scan again, or do a Binary Search.

#### Approach

* avoid Integer overflow
* let's use languages' APIs: `sumOf`, `indexOfFirst`, `position`
* in C++ let's implement the Binary Search

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin

    fun chalkReplacer(chalk: IntArray, k: Int): Int {
        var k = k.toLong() % chalk.sumOf { it.toLong() }
        return max(0, chalk.indexOfFirst { k -= it;  k < 0 })
    }

```
```rust

    pub fn chalk_replacer(chalk: Vec<i32>, k: i32) -> i32 {
       let mut k = k as i64 % chalk.iter().map(|&c| c as i64).sum::<i64>();
       chalk.iter().position(|&c| { k -= c as i64; k < 0 }).unwrap_or(0) as i32
    }

```
```c++

    int chalkReplacer(vector<int>& chalk, int k) {
        for (int i = 0; i < chalk.size(); i++) {
            if (i > 0) chalk[i] += chalk[i - 1];
            if (chalk[i] > k || chalk[i] < 0) return i;
        }
        k %= chalk[chalk.size() - 1];
        return upper_bound(chalk.begin(), chalk.end(), k) - chalk.begin();
    }

```

