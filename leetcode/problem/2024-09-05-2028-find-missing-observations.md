---
layout: leetcode-entry
title: "2028. Find Missing Observations"
permalink: "/leetcode/problem/2024-09-05-2028-find-missing-observations/"
leetcode_ui: true
entry_slug: "2024-09-05-2028-find-missing-observations"
---

[2028. Find Missing Observations](https://leetcode.com/problems/find-missing-observations/description/) medium
[blog post](https://leetcode.com/problems/find-missing-observations/solutions/5739995/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/05092024-2028-find-missing-observations?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/0-9QMzPHc04)
![1.webp](/assets/leetcode_daily_images/a9b9dd7c.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/725

#### Problem TLDR

Find `n` numbers to make `[n m]/(n+m)=mean` #medium #math

#### Intuition

This is a arithmetic problem:

```j

    // mean = (sum(m) + sum(n)) / (n + m)
    // 1 5 6    mean=3 n=4 m=3
    // 3 = ((1+5+6) + (x+y+z+k)) / (3+4)
    // 3*7 = 12 + ans
    // ans = 21 - 12 = 9
    // sum(ans) = 9, count(ans) = 4
    // 9 / 4 = 2
    //
    // 1 2 3 4 = 10 n=4 m=4 (n+m)=8 mean=6
    // mean*(n+m)=6*8=48
    // mean*(n+m)-sum = 48-10=38
    // (mean*(n+m)-sum)/n = 38/4 = [9 9 9 11]
    // 1 2 3 4 9 9 9 11    = 48 / 8 = 6 ???

```

The main trick is to not forget we only having the numbers `1..6`.

#### Approach

* we can check the numbers afterwards to be in `1..6` range
* the remainder is always less than `n`, so at most `1` can be added

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin

    fun missingRolls(rolls: IntArray, mean: Int, n: Int): IntArray {
        val x = mean * (n + rolls.size) - rolls.sum()
        return IntArray(n) { if (it < x % n) x / n + 1 else x / n }
            .takeIf { it.all { it in 1..6 }} ?: intArrayOf()
    }

```
```rust

    pub fn missing_rolls(rolls: Vec<i32>, mean: i32, n: i32) -> Vec<i32> {
        let x = mean * (n + rolls.len() as i32) - rolls.iter().sum::<i32>();
        if x < n || x > n * 6 { return vec![] }
        (0..n as usize).map(|i| x / n + (x % n > i as i32) as i32).collect()
    }

```
```c++

    vector<int> missingRolls(vector<int>& rolls, int mean, int n) {
        int x = mean * (n + rolls.size()) - accumulate(rolls.begin(), rolls.end(), 0);
        if (x < n || x > n * 6) return {};
        vector<int> res; for (int i = 0; i < n; i++) res.push_back(x / n + (x % n > i));
        return res;
    }

```

