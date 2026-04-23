---
layout: leetcode-entry
title: "1937. Maximum Number of Points with Cost"
permalink: "/leetcode/problem/2024-08-17-1937-maximum-number-of-points-with-cost/"
leetcode_ui: true
entry_slug: "2024-08-17-1937-maximum-number-of-points-with-cost"
---

[1937. Maximum Number of Points with Cost](https://leetcode.com/problems/maximum-number-of-points-with-cost/description/) medium
[blog post](https://leetcode.com/problems/maximum-number-of-points-with-cost/solutions/5648932/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/17082024-1937-maximum-number-of-points?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/9CVwO4CdxD0)
![1.webp](/assets/leetcode_daily_images/33bb9868.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/705

#### Problem TLDR

Max top-down path sum with column diff in 2D matrix #medium #dynamic_programming

#### Intuition

Let's observer all possible paths:
![2.webp](/assets/leetcode_daily_images/b08c144c.webp)

We only need the previous row, where each cell must be just a maximum of all incoming paths. For each cell we must check all the cells from the previous row. This will take O(row * col * row)

Let's observe how the maximum behaves when we walking on the `x` coordinate:
![3.webp](/assets/leetcode_daily_images/b386c6a6.webp)
As we see, the maximum is decreased each time by `one`, until it meets a bigger number. We can use this, but we lose all the right-to-left maximums, so let's walk both ways.

#### Approach

* we can store only two rows
* we can walk both directions in a single loop

#### Complexity

- Time complexity:
$$O(rc)$$

- Space complexity:
$$O(r)$$

#### Code

```kotlin

    fun maxPoints(points: Array<IntArray>): Long {
        var prev = LongArray(points[0].size); var curr = prev.clone()
        for (row in points) {
            var max = 0L; var max2 = 0L
            for (x in row.indices) {
                max--; max2--; val x2 = row.size - 1 - x
                max = max(max, prev[x]); max2 = max(max2, prev[x2])
                curr[x] = max(curr[x], row[x] + max)
                curr[x2] = max(curr[x2], row[x2] + max2)
            }
            prev = curr.also { curr = prev }
        }
        return prev.max()
    }

```
```rust

    pub fn max_points(points: Vec<Vec<i32>>) -> i64 {
        let (mut dp, mut i, mut res) = (vec![vec![0; points[0].len()]; 2], 0, 0);
        for row in points {
            let (mut max, mut max2) = (0, 0);
            for x in 0..row.len() {
                max -= 1; max2 -= 1; let x2 = row.len() - 1 - x;
                max = max.max(dp[i][x]); max2 = max2.max(dp[i][x2]);
                dp[1 - i][x] = dp[1 - i][x].max(row[x] as i64 + max);
                dp[1 - i][x2] = dp[1 - i][x2].max(row[x2] as i64 + max2);
                res = res.max(dp[1 - i][x]).max(dp[1 - i][x2])
            }
            i = 1 - i
        }; res
    }

```

