---
layout: leetcode-entry
title: "1353. Maximum Number of Events That Can Be Attended"
permalink: "/leetcode/problem/2025-07-07-1353-maximum-number-of-events-that-can-be-attended/"
leetcode_ui: true
entry_slug: "2025-07-07-1353-maximum-number-of-events-that-can-be-attended"
---

[1353. Maximum Number of Events That Can Be Attended](https://leetcode.com/problems/maximum-number-of-events-that-can-be-attended/description/) medium
[blog post](https://leetcode.com/problems/maximum-number-of-events-that-can-be-attended/solutions/6930245/kotlin-rust-by-samoylenkodmitry-zgjl/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/7072025-1353-maximum-number-of-events?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/6nPapN6xo2s)
![1.webp](/assets/leetcode_daily_images/1b24a096.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1042

#### Problem TLDR

Max attended events #medium #heap

#### Intuition

```j

    //  1  2  3  4
    //  ****
    //  ****
    //     ****
    //        ****
    //
    //  1  2  3  4
    //  **********
    //           *
    //     *
    //        ****
    //  *
    //
    //  1  2  3  4  5
    //  1************
    //  *********4***
    //  ************5
    //     2***
    //     ***3
    //  3  5  5  3  3
    //  2  4  4  2  2  take 1
    //     3  3  2  2  take 1 (until it's end)
    //        2  2  2  take 1 (until it's end)
    //           1  1  take 1 (until it's end, search for end)
    //              0  take 1
    //
    // 1 2 3 4 5 6 7
    // *
    // ***
    // *****
    // *******
    // *********
    // ***********
    // *************

```
The greedy strategy is to never waste a day and prioritize those that ends soon.
* iterate over days
* close already ended
* add all started in that day
* take one that ends sooner (maintain a heap to take min)

#### Approach

* iteration over days range is almost as fast as manual day adjusting

#### Complexity

- Time complexity:
$$O(nlog(n))$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin

// 97ms
    fun maxEvents(es: Array<IntArray>): Int {
        val days = Array(100002) { ArrayList<Int>() }
        for ((s, e) in es) days[s] += e
        var cnt = 0; val pq = PriorityQueue<Int>()
        for (d in 0..100000) {
            while (pq.size > 0 && pq.peek() < d) pq.poll()
            pq += days[d]
            if (pq.size > 0) { pq.poll(); cnt++ }
        }
        return cnt
    }

```
```kotlin

// 96ms
    fun maxEvents(es: Array<IntArray>): Int {
        es.sortWith(compareBy({ it[0] }, { it[1] }))
        val pq = PriorityQueue<Int>()
        var d = 0; var i = 0; var cnt = 0
        for (d in 1..100000) {
            while (pq.size > 0 && pq.peek() < d) pq.poll()
            while (i < es.size && es[i][0] == d) pq += es[i++][1]
            if (pq.size > 0) { pq.poll(); ++cnt }
        }
        return cnt
    }

```
```kotlin

// 94ms
    fun maxEvents(es: Array<IntArray>): Int {
        es.sortWith(compareBy({ it[0] }, { it[1] }))
        val pq = PriorityQueue<Int>()
        var d = 0; var i = 0; var cnt = 0
        while (pq.size > 0 || i < es.size) {
            if (pq.size < 1) d = es[i][0]
            while (i < es.size && es[i][0] == d) pq += es[i++][1]
            pq.poll(); ++cnt; ++d
            while (pq.size > 0 && pq.peek() < d) pq.poll()
        }
        return cnt
    }

```
```rust

// 24ms
    pub fn max_events(mut es: Vec<Vec<i32>>) -> i32 {
        es.sort_unstable();
        let (mut pq, mut i, mut cnt) = (BinaryHeap::new(), 0, 0);
        for d in 1..=100000 {
            while pq.len() > 0 && -pq.peek().unwrap() < d { pq.pop(); }
            while i < es.len() && es[i][0] == d { pq.push(-es[i][1]); i += 1 }
            if let Some(_) = pq.pop() { cnt += 1 }
        } cnt
    }

```
```c++

// 69ms
    int maxEvents(vector<vector<int>>& es) {
        sort(begin(es), end(es));
        priority_queue<int, vector<int>, greater<int>> pq;
        int i = 0, cnt = 0;
        for (int d = 1; d <= 100000; ++d) {
            while (size(pq) && pq.top() < d) pq.pop();
            while (i < size(es) && es[i][0] == d) pq.push(es[i++][1]);
            if (size(pq)) { pq.pop(); ++cnt; }
        } return cnt;
    }

```

