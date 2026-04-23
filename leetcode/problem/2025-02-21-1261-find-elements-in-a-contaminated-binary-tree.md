---
layout: leetcode-entry
title: "1261. Find Elements in a Contaminated Binary Tree"
permalink: "/leetcode/problem/2025-02-21-1261-find-elements-in-a-contaminated-binary-tree/"
leetcode_ui: true
entry_slug: "2025-02-21-1261-find-elements-in-a-contaminated-binary-tree"
---

[1261. Find Elements in a Contaminated Binary Tree](https://leetcode.com/problems/find-elements-in-a-contaminated-binary-tree/description/) medium
[blog post](https://leetcode.com/problems/find-elements-in-a-contaminated-binary-tree/solutions/6450115/kotlin-rust-by-samoylenkodmitry-0k70/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/21022025-1261-find-elements-in-a?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/gkbQqSvehJg)
![1.webp](/assets/leetcode_daily_images/9bc60121.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/902

#### Problem TLDR

Is number in a tree 2*v+(1L, 2R)? #medium #bfs #dfs

#### Intuition

Collect values into a HashSet, then check.
With Breadth-First Search values are always growing, we can use a BinarySearch.

We can also restore the original path for each value:

```j

 //           0
 //    1                 2
 // 3     4           5      6
 //7 8  9  10      11   12 13  14
 // 0, 1..2, 3..6, 7..14, 15..30,

 // 13(L) -> (13 - 1) / 2 = 6(R), (6-2)/2 = 2, (2-2) / 2 = 0

```

#### Approach

* let's implement DFS + HashSet, BFS + BinarySearch, and a path walking solutions

#### Complexity

- Time complexity:
$$O(n)$$, or O(nlog(n)) for the path walk or the binary search

- Space complexity:
$$O(n)$$, or O(1) for the path walk

#### Code

```kotlin

class FindElements(root: TreeNode?): HashSet<Int>() {
    fun dfs(x: TreeNode?, v: Int): Unit? =
      x?.run {add(v); dfs(x?.left, 2 * v + 1); dfs(x?.right, 2 * v + 2)}
    init { dfs(root, 0) }
    fun find(target: Int) = target in this
}

```
```rust

struct FindElements(Option<Rc<RefCell<TreeNode>>>);
impl FindElements {
    fn new(root: Option<Rc<RefCell<TreeNode>>>) -> Self { Self(root) }
    fn find(&self, target: i32) -> bool {
        let (mut x, mut path, mut n) = (target, vec![], self.0.clone());
        while x > 0 { path.push(x % 2); x = (x - 2 + (x % 2)) / 2 }
        for i in (0..path.len()).rev() {
            let Some(m) = n else { return false }; let m = m.borrow();
            n = if path[i] > 0 { m.left.clone() } else { m.right.clone() }
        }; n.is_some()
    }
}

```
```c++

class FindElements {
public: vector<int> s;
    FindElements(TreeNode* root) {
        for (queue<pair<TreeNode*, int>> q({ {root, 0} }); size(q);) {
            auto [n, v] = q.front(); q.pop();
            if (n) s.push_back(v), q.push({n->left, v * 2 + 1}), q.push({n->right, v * 2 + 2});
        }
    }
    bool find(int target) { return std::binary_search(begin(s), end(s), target); }
};

```

