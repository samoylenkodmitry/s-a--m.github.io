---
layout: leetcode-entry
title: "3362. Zero Array Transformation III"
permalink: "/leetcode/problem/2025-05-22-3362-zero-array-transformation-iii/"
leetcode_ui: true
entry_slug: "2025-05-22-3362-zero-array-transformation-iii"
---

[3362. Zero Array Transformation III](https://leetcode.com/problems/zero-array-transformation-iii/description) medium
[blog post](https://leetcode.com/problems/zero-array-transformation-iii/solutions/6769290/kotlin-rust-by-samoylenkodmitry-bkte/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/22052025-3362-zero-array-transformation?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/t3khX-ZiZfI)
![1.webp](/assets/leetcode_daily_images/8be00294.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/996

#### Problem TLDR

Max removed queries still zero-fy an array #medium #heap

#### Intuition

Didn't solved.

Irrelevant chain-of-thougths for history:
```j

    // [1,1,1,1]
    // [[1,3],[0,2],[1,3],[1,2]]
    // 0,2 1,2 1,3 1,3
    // 0  1  2  3
    // 0, 3, 3, 1, -2
    //    i  j           0,2 take
    //                   1,2 drop, 3,3 -> 2,2 min=2
    //    i     j        1,3 drop, 2,2,1 -> 2,2,0, min=0
    // *running interval minimum* increasing queue?
    // looks like too hard for medium, maybe wrong algo?
    // use hints
    // sort: already done
    // pick max end: already do ?
    //
    // [1,1,1,1]
    //        i    1..3
    //  i          max = 1
    //             1..3
    //   2 0 2
    //   1   1    0..2
    //
    //   2   2    0..2
    //     1      1..1 move and compute running sum
    // ok, i fail

```

The working solution:
* sort queries by start
* iterate the nums
* maintain the current accepted queries sum
* put accepted queries (start, end) into a line sweep diff array
* *hard part*: put candidate queries (same start) ends into a sorted heap, poll lazily when needed

#### Approach

* it looks like I've solved it previously in contest, but havn't absorbed the solution, or, even possible degraded in the solution search
* I've solved the wrong problem on the start and spent some mind power

#### Complexity

- Time complexity:
$$O(nlog(n))$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin

    fun maxRemoval(nums: IntArray, queries: Array<IntArray>): Int {
        val h = PriorityQueue<Int>(); val d = IntArray(nums.size + 1)
        queries.sortBy { it[0] }; var qsum = 0; var j = 0
        for ((i, x) in nums.withIndex()) {
            while (j < queries.size && queries[j][0] == i) h += -queries[j++][1]
            qsum += d[i]
            while (x > qsum && h.size > 0 && -h.peek() >= i) {
                d[-h.poll() + 1]--; qsum++
            }
            if (x > qsum) return -1
        }
        return h.size
    }

```
```rust

    pub fn max_removal(n: Vec<i32>, mut q: Vec<Vec<i32>>) -> i32 {
        let (mut h, mut d) = (BinaryHeap::new(), vec![0; n.len() + 1]);
        q.sort_unstable(); let (mut lvl, mut j) = (0, 0);
        for i in 0..n.len() {
            while j < q.len() && q[j][0] == i as i32 { h.push(q[j][1] as usize); j += 1 }
            lvl += d[i];
            while n[i] > lvl && h.len() > 0 && *h.peek().unwrap() >= i {
                d[h.pop().unwrap() + 1] -= 1; lvl += 1
            }
            if n[i] > lvl { return -1 }
        } h.len() as _
    }

```
```c++

    int maxRemoval(vector<int>& n, vector<vector<int>>& q) {
        priority_queue<int> h; int l = 0, j = 0;
        vector<int> d(n.size() + 1); sort(q.begin(), q.end());
        for (int i = 0, N = n.size(); i < N; ++i) {
            while (j < q.size() && q[j][0] == i)
                h.push(q[j++][1]);
            l += d[i];
            while (l < n[i] && !h.empty() && h.top() >= i)
                l++, d[h.top() + 1]--, h.pop();
            if (l < n[i]) return -1;
        }
        return h.size();
    }

```

