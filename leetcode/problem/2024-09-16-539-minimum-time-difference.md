---
layout: leetcode-entry
title: "539. Minimum Time Difference"
permalink: "/leetcode/problem/2024-09-16-539-minimum-time-difference/"
leetcode_ui: true
entry_slug: "2024-09-16-539-minimum-time-difference"
---

[539. Minimum Time Difference](https://leetcode.com/problems/minimum-time-difference/description/) medium
[blog post](https://leetcode.com/problems/minimum-time-difference/solutions/5793485/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/16092024-539-minimum-time-difference?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/NcooyRfl2xU)
![1.webp](/assets/leetcode_daily_images/4965e91a.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/737

#### Problem TLDR

Min difference in a list of times `hh:mm` #medium

#### Intuition

The main problem is how to handle the loop:

```j

    // 12:00
    //  1:00
    // 23:00

```
One way is to repeat the array twice. (Actually, only the first value matters).

#### Approach

* let's use `window` iterator
* we can use a bucket sort

#### Complexity

- Time complexity:
$$O(nlog(n))$$ or $$O(n + m)$$, where m is minutes = 24 * 60

- Space complexity:
$$O(n)$$ or $$O(m)$$

#### Code

```kotlin

    fun findMinDifference(timePoints: List<String>) =
        timePoints.map { it.split(":")
            .let { it[0].toInt() * 60 + it[1].toInt() }}
        .sorted().let { it + (it[0] + 24 * 60) }
        .windowed(2).minOf { it[1] - it[0] }

```
```rust

    pub fn find_min_difference(time_points: Vec<String>) -> i32 {
        let mut times: Vec<_> = time_points.iter().map(|s| {
            s[0..2].parse::<i32>().unwrap() * 60 + s[3..5].parse::<i32>().unwrap()
        }).collect();
        times.sort_unstable(); times.push(times[0] + 60 * 24);
        times.windows(2).map(|w| w[1] - w[0]).min().unwrap()
    }

```
```c++

    int findMinDifference(vector<string>& timePoints) {
        vector<bool> times(24 * 60 + 1);
        for (int i = 0; i < timePoints.size(); i++) {
            int t = std::stoi(timePoints[i].substr(0, 2)) * 60 +
                std::stoi(timePoints[i].substr(3, 5));
            if (times[t]) return 0; else times[t] = true;
        }
        int res = times.size(); int j = -1; int first = -1;
        for (int i = 0; i < times.size(); i++) if (times[i]) {
            if (j >= 0) res = min(res, i - j); else first = i;
            j = i;
        }
        return min(res, first + 24 * 60 - j);
    }

```

