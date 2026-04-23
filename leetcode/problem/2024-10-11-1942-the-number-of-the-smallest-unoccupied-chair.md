---
layout: leetcode-entry
title: "1942. The Number of the Smallest Unoccupied Chair"
permalink: "/leetcode/problem/2024-10-11-1942-the-number-of-the-smallest-unoccupied-chair/"
leetcode_ui: true
entry_slug: "2024-10-11-1942-the-number-of-the-smallest-unoccupied-chair"
---

[1942. The Number of the Smallest Unoccupied Chair](https://leetcode.com/problems/the-number-of-the-smallest-unoccupied-chair/description/) medium
[blog post](https://leetcode.com/problems/the-number-of-the-smallest-unoccupied-chair/solutions/5898579/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/11102024-1942-the-number-of-the-smallest?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/MucA5ws2d6s)
![1.webp](/assets/leetcode_daily_images/73b0de50.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/765

#### Problem TLDR

`target's` chair number when chairs reused by multiple `[arrival, leave]` times #medium #sorting #heap

##### Intuition

Let's observe what we can do with those intervals:

```j

    // 3,10  1,5  2,6  6,7     t=3
    // 1 2 3 4 5 6 7 8 9 10
    //
    //     0 0 0 0 0 0 0 0   2
    // 1 1 1 1 1             0
    //   2 2 2 2 2           1
    //           3 3         0

```

The line sweep technique will work here: first sort split each interval into two events `arrival` and `leave`, then sort and iterate.

To keep track of the chairs, let's use some sorted collection: TreeSet or Heap (PriorityQueue).

#### Approach

* no more than `times.size` chairs total
* sort by the `leave` first to free the chair before arrival at the same time

#### Complexity

- Time complexity:
$$O(nlog(n))$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin

    fun smallestChair(times: Array<IntArray>, targetFriend: Int): Int {
        val free = PriorityQueue<Int>(); val iToChair = mutableMapOf<Int, Int>()
        val inds = mutableListOf<Pair<Int, Int>>()
        for (i in times.indices) { inds += i to 0; inds += i to 1; free += i }
        inds.sortWith(compareBy({ times[it.first][it.second] }, { -it.second }))
        for ((i, t) in inds) if (t == 1) free += iToChair.remove(i)!! else {
            iToChair[i] = free.poll()
            if (i == targetFriend) return iToChair[i]!!
        }
        return -1
    }

```
```rust

    pub fn smallest_chair(times: Vec<Vec<i32>>, target_friend: i32) -> i32 {
        let (mut free, mut i_to_chair, mut inds) = (BinaryHeap::new(), HashMap::new(), vec![]);
        for i in 0..times.len() { inds.push((i, 0)); inds.push((i, 1)); free.push(-(i as i32)); }
        inds.sort_unstable_by_key(|&(i, t)| (times[i][t], -(t as i32)));
        for (i, t) in inds { if t == 0 {
            i_to_chair.insert(i, -free.pop().unwrap());
            if target_friend == i as i32 { return i_to_chair[&i]; }
        } else { free.push(-i_to_chair[&i]); }}; -1
    }

```
```c++

    int smallestChair(vector<vector<int>>& times, int targetFriend) {
        vector<array<int, 3>> e; set<int> free; vector<int> seated(times.size());
        for (int i = 0; i < times.size(); ++i)
            e.push_back({times[i][0], 0, i}), e.push_back({times[i][1], -1, i}), free.insert(i);
        sort(e.begin(), e.end());
        for (auto [_, l, p] : e) if (l) free.insert(seated[p]); else {
            seated[p] = *free.begin();
            free.erase(free.begin());
            if (p == targetFriend) return seated[p];
        }
        return -1;
    }

```

