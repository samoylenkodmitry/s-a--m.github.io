---
layout: leetcode-entry
title: "2054. Two Best Non-Overlapping Events"
permalink: "/leetcode/problem/2025-12-23-2054-two-best-non-overlapping-events/"
leetcode_ui: true
entry_slug: "2025-12-23-2054-two-best-non-overlapping-events"
---

[2054. Two Best Non-Overlapping Events](https://leetcode.com/problems/two-best-non-overlapping-events/description) medium
[blog post](https://leetcode.com/problems/two-best-non-overlapping-events/solutions/7432722/kotlin-rust-by-samoylenkodmitry-lr27/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/23122025-2054-two-best-non-overlapping?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/INZVk7nnfc4)

![f7189f0f-5937-475d-86fd-2483ce267166 (1).webp](/assets/leetcode_daily_images/f98f0426.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1213

#### Problem TLDR

Max at most two non-overlapping events #medium #prefix_max #bs #pq

#### Intuition

```j
    // ************
    // *****
    //  *****
    //   *****
    //    *****
    //     *****
    //     .*****
    //     . *****
    //
    // need a way to map[time][maxv]
```

* One way: sort by starts, put visited into PriorityQueue, poll by ends
* Another way: sort by ends, put visited into prefix-max list, binary search start in ends

#### Approach

* the prefix-max requires initial (0,0) value
* there is also a suffix-max solution

#### Complexity

- Time complexity:
$$O(nlogn)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin
// 135ms
    fun maxTwoEvents(e: Array<IntArray>): Int {
        var max = 0; val q = PriorityQueue<Int>(compareBy{e[it][1]})
        return e.indices.sortedBy { e[it][0]}.maxOf { i ->
            while (q.size > 0 && e[q.first()][1] < e[i][0])
                max = max(max, e[q.poll()][2])
            q += i; e[i][2] + max
        }
    }
```
```rust
// 10ms
    pub fn max_two_events(mut e: Vec<Vec<i32>>) -> i32 {
        let mut max = vec![(0,0)]; e.sort_unstable_by_key(|e|e[1]);
        e.iter().map(|e|{
            max.push((max[max.len()-1].0.max(e[2]), e[1]));
            e[2] + max[max.partition_point(|m| m.1 < e[0]) - 1].0
        }).max().unwrap()
    }
```

