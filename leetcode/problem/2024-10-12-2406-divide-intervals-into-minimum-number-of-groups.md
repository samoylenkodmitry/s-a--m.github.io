---
layout: leetcode-entry
title: "2406. Divide Intervals Into Minimum Number of Groups"
permalink: "/leetcode/problem/2024-10-12-2406-divide-intervals-into-minimum-number-of-groups/"
leetcode_ui: true
entry_slug: "2024-10-12-2406-divide-intervals-into-minimum-number-of-groups"
---

[2406. Divide Intervals Into Minimum Number of Groups](https://leetcode.com/problems/divide-intervals-into-minimum-number-of-groups/description/) medium
[blog post](https://leetcode.com/problems/divide-intervals-into-minimum-number-of-groups/solutions/5902666/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/12102024-2406-divide-intervals-into?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/A_Aq8b5uGj0)
![1.webp](/assets/leetcode_daily_images/6bdd537e.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/766

#### Problem TLDR

Count non-intersecting groups of intervals #medium #heap #sorting

#### Intuition

Let's observe the intervals' properties:

```j

    // 5,10 6,8 1,5 2,3 1,10   n=5
    //
    // 1 2 3 4 5 6 7 8 9 10
    //         . . . . . .  5->take min g3(10)
    //           . . .      6->take min g1(5)
    // . . . . .             g1(5)
    //   . .                 g3(3)
    // . . . . . . . . . .   g2(10)

```

If we use `sweep line` algorithm, then we should peek the first non-intersecting group or add another group. To track the groups, let's maintain a `heap` with `ends` of each group.

Another way to solve this is to notice some observation: `groups count is the maximum intersecting intervals count` (but it should be proved somehow, it is just works magically)

#### Approach

* to sweep line to work, we should sort for both `ends` and `starts` to be in increasing order
* for the second way, we can use counting sorting

#### Complexity

- Time complexity:
$$O(n)$$ or O(nlog(n))

- Space complexity:
$$O(m)$$ or O(n)

#### Code

```kotlin

    fun minGroups(intervals: Array<IntArray>): Int {
        intervals.sortWith(compareBy({ it[0] }, { it[1] }))
        val ends = PriorityQueue<Int>()
        for ((a, b) in intervals) {
            if (ends.size > 0 && a > ends.peek()) ends.poll()
            ends += b
        }
        return ends.size
    }

```
```rust

    pub fn min_groups(intervals: Vec<Vec<i32>>) -> i32 {
        let (mut ends, mut curr) = (vec![0; 1_000_002], 0);
        for iv in intervals {
            ends[iv[0] as usize] += 1; ends[1 + iv[1] as usize] -= 1; }
        ends.iter().map(|e| { curr += e; curr }).max().unwrap()
    }

```
```c++

    int minGroups(vector<vector<int>>& intervals) {
        int ends[1000002] = {}, curr = 0, res = 0;
        for (auto iv: intervals) ends[iv[0]]++, ends[iv[1] + 1]--;
        for (int e: ends) res = max(res, curr += e);
        return res;
    }

```

