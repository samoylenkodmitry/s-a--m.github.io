---
layout: leetcode-entry
title: "2054. Two Best Non-Overlapping Events"
permalink: "/leetcode/problem/2024-12-08-2054-two-best-non-overlapping-events/"
leetcode_ui: true
entry_slug: "2024-12-08-2054-two-best-non-overlapping-events"
---

[2054. Two Best Non-Overlapping Events](https://leetcode.com/problems/two-best-non-overlapping-events/description/) medium
[blog post](https://leetcode.com/problems/two-best-non-overlapping-events/solutions/6125512/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/08122024-2054-two-best-non-overlapping?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/uBoGo4P9t9E)
[deep-dive](https://notebooklm.google.com/notebook/a8982576-4151-42ec-8693-42faa48bf158/audio)
![1.webp](/assets/leetcode_daily_images/6f988dcf.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/826

#### Problem TLDR

Max two non-overlapping intervals #medium #binary_search

#### Intuition

Let's observe some rich example and try to invent an algorithm:

```j

    // 0123456789111
    //    .      012
    // [.......7...]
    // [.3][.2]   .
    //  [.5][.3]  .
    //   [.4][.4] .
    //    [.2][.6].
    //    .[.1][.7]
    //    ..  .   .
    //    t.  .   .   t=3, v=3 maxV=3 maxT=3
    //     t  .   .   t=4, v=5,maxV=5 maxT=4
    //     .t .   .   t=5, v=4,
    //     . t.   .   t=6, v=2
    //     .  t   .   t=7, v=2
    //     .  t   .   t=7, v=1
    //     .  .t  .   t=8, v=3
    //     .  . t .   t=9, v=4
    //     .  .  t.   t=10,v=6, maxV=6, maxT=10
    //     .  .  .t   t=11,v=7, maxV=7, maxT=11
    //     .  .  . t  t=12,v=7
    //    3555555677  maxV
    //        *f  t   5+7

```
Some observations:
* for current interval we should find the maximum before it
* we can store the maximums as we go
* we should sort events by the `end` times

Another two approaches:
1. use a Heap, sort by start, pop from heap all non-intersecting previous and peek a max
2. line sweep: put starts and ends in a timeline, sort by time, compute `max` after ends, and `res` on start

#### Approach

* binary search approach have many subtle tricks: add (0, 0) as zero, sort also by bigger values first to make binary search work

#### Complexity

- Time complexity:
$$O(nlog(n))$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin

    fun maxTwoEvents(events: Array<IntArray>): Int {
        val pq = PriorityQueue<Pair<Int, Int>>(compareBy { it.first })
        var res = 0; var max = 0;
        for ((f, t, v) in events.sortedBy { it[0] }) {
            while (pq.size > 0 && pq.peek().first < f)
                max = max(max, pq.poll().second)
            res = max(res, max + v)
            pq += t to v
        }
        return res
    }

```
```rust

    pub fn max_two_events(mut events: Vec<Vec<i32>>) -> i32 {
        let (mut res, mut m) = (0, vec![(0, 0)]);
        events.sort_unstable_by_key(|e| (e[1], -e[2]));
        for e in events {
            let i = m.partition_point(|x| x.1 < e[0]) - 1;
            m.push((m.last().unwrap().0.max(e[2]), e[1]));
            res = res.max(m[i].0 + e[2]);
        }; res
    }

```
```c++

    int maxTwoEvents(vector<vector<int>>& events) {
        vector<tuple<int, int, int>> t; int res = 0, m = 0;
        for (auto e: events)
            t.push_back({e[0], 1, e[2]}),
            t.push_back({e[1] + 1, 0, e[2]});
        sort(begin(t), end(t));
        for (auto [x, start, v]: t)
            start ? res = max(res, m + v) : m = max(m, v);
        return res;
    }

```

