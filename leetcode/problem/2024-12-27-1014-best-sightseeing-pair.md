---
layout: leetcode-entry
title: "1014. Best Sightseeing Pair"
permalink: "/leetcode/problem/2024-12-27-1014-best-sightseeing-pair/"
leetcode_ui: true
entry_slug: "2024-12-27-1014-best-sightseeing-pair"
---

[1014. Best Sightseeing Pair](https://leetcode.com/problems/best-sightseeing-pair/description/) medium
[blog post](https://leetcode.com/problems/best-sightseeing-pair/solutions/6192314/kotlin-rust-by-samoylenkodmitry-t9lj/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/27122024-1014-best-sightseeing-pair?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/hHhpqPY6DZU)
[deep-dive](https://notebooklm.google.com/notebook/42c0baee-e59d-40af-aba8-421349b5397d/audio)
![1.webp](/assets/leetcode_daily_images/8c898f3b.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/845

#### Problem TLDR

Max (a[i] + a[j] + i - j), i > j #medium #arithmetics

#### Intuition

Let's move the pointers and observe:

```j

    // 0 1 2 3
    // 3 1 2 5
    // j
    //       i    5 - (3 - 0) + 3 = 5 - 3   +  0 + 3
    //            5 - (3 - 1) + 1 = 5 - 3   +  1 + 1
    //            5 - (3 - 2) + 2 = 5 - 3   +  2 + 2

```

Each time we move `i`, all possible previous sums are decreased by distance of `1`. By writing down `a[i] - (i - j) + a[j]` in another way: `(a[i] - i) + (a[j] + j)` we derive the total sum is independent of the distance, always peek the max of `a[j] + j` from the previous.

Some other things I've considered are: sorting, monotonic stack. But didn't see any good use of them.

#### Approach

* the first previous value can be `0` instead of `values[0]`

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin

    fun maxScoreSightseeingPair(values: IntArray): Int {
        var p = 0
        return values.withIndex().maxOf { (i, n) ->
            (n - i + p).also { p = max(p, i + n) }
        }
    }

```
```rust

    pub fn max_score_sightseeing_pair(values: Vec<i32>) -> i32 {
        let mut p = 0;
        values.iter().enumerate().map(|(i, n)| {
            let c = n - i as i32 + p; p = p.max(i as i32 + n); c
        }).max().unwrap()
    }

```
```c++

    int maxScoreSightseeingPair(vector<int>& values) {
        int res = 0;
        for (int i = 0, p = 0; i < values.size(); ++i)
            res = max(res, values[i] - i + p), p = max(p, values[i] + i);
        return res;
    }

```

