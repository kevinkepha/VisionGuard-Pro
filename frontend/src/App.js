import React, { useState, useEffect, useCallback } from 'react';
import {
  AppBar, Toolbar, Typography, Drawer, List, ListItem, ListItemIcon, ListItemText,
  Card, CardContent, Grid, Box, Button, Dialog, DialogTitle, DialogContent,
  DialogActions, TextField, Table, TableBody, TableCell, TableContainer,
  TableHead, TableRow, Paper, Chip, LinearProgress, IconButton, Menu,
  MenuItem, Alert, Snackbar, Badge, Tabs, Tab, CircularProgress,
  FormControl, InputLabel, Select, Avatar, Divider, Switch, FormControlLabel
} from '@mui/material';
import {
  Dashboard, PhotoCamera, Analytics, Settings, Upload, Warning,
  CheckCircle, Error, Timeline, Build, Notifications, CloudUpload,
  FileDownload, Refresh, MoreVert, TrendingUp, Speed, Memory,
  Security, Storage, PlayArrow, Stop, Visibility, VisibilityOff
} from '@mui/icons-material';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, 
  ResponsiveContainer, BarChart, Bar, PieChart, Pie, Cell, AreaChart, Area } from 'recharts';

const API_BASE = 'http://localhost:8000';
const DEMO_TOKEN = 'demo-token';

// Mock data for demo
const mockAnalytics = {
  summary: {
    total_inspections: 1247,
    avg_quality_score: 0.892,
    total_defects: 89,
    avg_processing_time: 1.34,
    defect_rate: 0.071
  },
  defect_breakdown: [
    { type: 'scratch', count: 34, avg_confidence: 0.87 },
    { type: 'dent', count: 28, avg_confidence: 0.91 },
    { type: 'discoloration', count: 15, avg_confidence: 0.83 },
    { type: 'crack', count: 8, avg_confidence: 0.94 },
    { type: 'contamination', count: 4, avg_confidence: 0.78 }
  ],
  daily_trends: [
    { date: '2025-08-25', inspections: 178, avg_quality: 0.89, defects: 12 },
    { date: '2025-08-26', inspections: 192, avg_quality: 0.91, defects: 8 },
    { date: '2025-08-27', inspections: 156, avg_quality: 0.87, defects: 15 },
    { date: '2025-08-28', inspections: 203, avg_quality: 0.93, defects: 6 },
    { date: '2025-08-29', inspections: 185, avg_quality: 0.88, defects: 11 },
    { date: '2025-08-30', inspections: 167, avg_quality: 0.90, defects: 9 },
    { date: '2025-08-31', inspections: 166, avg_quality: 0.89, defects: 13 }
  ]
};

const mockInspections = [
  {
    id: '1',
    product_line: 'Automotive Line A',
    timestamp: '2025-08-31T10:30:00Z',
    quality_score: 0.92,
    defects_detected: 0,
    status: 'completed'
  },
  {
    id: '2',
    product_line: 'Electronics Line B',
    timestamp: '2025-08-31T10:25:00Z',
    quality_score: 0.76,
    defects_detected: 2,
    status: 'completed'
  },
  {
    id: '3',
    product_line: 'Automotive Line A',
    timestamp: '2025-08-31T10:20:00Z',
    quality_score: 0.88,
    defects_detected: 1,
    status: 'completed'
  }
];

const mockProductLines = [
  { id: '1', name: 'Automotive Line A', quality_threshold: 0.85, active: true },
  { id: '2', name: 'Electronics Line B', quality_threshold: 0.90, active: true },
  { id: '3', name: 'Packaging Line C', quality_threshold: 0.80, active: false }
];

// Utility functions
const formatDate = (dateString) => {
  return new Date(dateString).toLocaleString();
};

const getStatusColor = (score) => {
  if (score >= 0.9) return 'success';
  if (score >= 0.8) return 'warning';
  return 'error';
};

const getStatusIcon = (score) => {
  if (score >= 0.9) return <CheckCircle color="success" />;
  if (score >= 0.8) return <Warning color="warning" />;
  return <Error color="error" />;
};

// Main Dashboard Component
const Dashboard = () => {
  const [selectedTab, setSelectedTab] = useState(0);
  const [drawerOpen, setDrawerOpen] = useState(true);
  const [analytics, setAnalytics] = useState(mockAnalytics);
  const [inspections, setInspections] = useState(mockInspections);
  const [productLines, setProductLines] = useState(mockProductLines);
  const [loading, setLoading] = useState(false);
  const [snackbar, setSnackbar] = useState({ open: false, message: '', severity: 'info' });
  const [uploadDialog, setUploadDialog] = useState(false);
  const [realTimeMode, setRealTimeMode] = useState(false);
  const [selectedProductLine, setSelectedProductLine] = useState('all');

  // Dashboard stats cards
  const StatsCard = ({ title, value, subtitle, icon, color = 'primary' }) => (
    <Card sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      <CardContent sx={{ flex: 1 }}>
        <Box display="flex" alignItems="center" justifyContent="space-between">
          <Box>
            <Typography color="textSecondary" gutterBottom variant="h6">
              {title}
            </Typography>
            <Typography variant="h3" component="div" color={color}>
              {value}
            </Typography>
            <Typography color="textSecondary" variant="body2">
              {subtitle}
            </Typography>
          </Box>
          <Avatar sx={{ bgcolor: `${color}.main`, width: 56, height: 56 }}>
            {icon}
          </Avatar>
        </Box>
      </CardContent>
    </Card>
  );

  // Quality trends chart
  const QualityTrendsChart = () => (
    <Card sx={{ height: 400 }}>
      <CardContent>
        <Typography variant="h6" gutterBottom>
          Quality Trends (Last 7 Days)
        </Typography>
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={analytics.daily_trends}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="date" />
            <YAxis />
            <Tooltip />
            <Legend />
            <Line type="monotone" dataKey="avg_quality" stroke="#1976d2" strokeWidth={3} />
            <Line type="monotone" dataKey="defects" stroke="#d32f2f" strokeWidth={2} />
          </LineChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );

  // Defect breakdown chart
  const DefectBreakdownChart = () => {
    const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884D8'];
    
    return (
      <Card sx={{ height: 400 }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Defect Types Distribution
          </Typography>
          <ResponsiveContainer width="100%" height={300}>
            <PieChart>
              <Pie
                data={analytics.defect_breakdown}
                cx="50%"
                cy="50%"
                labelLine={false}
                label={({ type, count }) => `${type}: ${count}`}
                outerRadius={80}
                fill="#8884d8"
                dataKey="count"
              >
                {analytics.defect_breakdown.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                ))}
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>
    );
  };

  // Recent inspections table
  const InspectionsTable = () => (
    <Card>
      <CardContent>
        <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
          <Typography variant="h6">Recent Inspections</Typography>
          <Button startIcon={<Refresh />} onClick={() => setLoading(true)}>
            Refresh
          </Button>
        </Box>
        <TableContainer>
          <Table>
            <TableHead>
              <TableRow>
                <TableCell>ID</TableCell>
                <TableCell>Product Line</TableCell>
                <TableCell>Quality Score</TableCell>
                <TableCell>Defects</TableCell>
                <TableCell>Timestamp</TableCell>
                <TableCell>Status</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {inspections.map((inspection) => (
                <TableRow key={inspection.id} hover>
                  <TableCell>{inspection.id}</TableCell>
                  <TableCell>{inspection.product_line}</TableCell>
                  <TableCell>
                    <Box display="flex" alignItems="center">
                      {getStatusIcon(inspection.quality_score)}
                      <Typography variant="body2" sx={{ ml: 1 }}>
                        {(inspection.quality_score * 100).toFixed(1)}%
                      </Typography>
                    </Box>
                  </TableCell>
                  <TableCell>
                    <Chip 
                      label={inspection.defects_detected}
                      color={inspection.defects_detected === 0 ? 'success' : 'error'}
                      size="small"
                    />
                  </TableCell>
                  <TableCell>{formatDate(inspection.timestamp)}</TableCell>
                  <TableCell>
                    <Chip 
                      label={inspection.status}
                      color="success"
                      variant="outlined"
                      size="small"
                    />
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </TableContainer>
      </CardContent>
    </Card>
  );

  // Upload dialog component
  const UploadDialog = () => {
    const [selectedFiles, setSelectedFiles] = useState([]);
    const [uploadProgress, setUploadProgress] = useState(0);
    const [uploading, setUploading] = useState(false);

    const handleFileSelect = (event) => {
      setSelectedFiles(Array.from(event.target.files));
    };

    const handleUpload = async () => {
      setUploading(true);
      // Simulate upload progress
      for (let i = 0; i <= 100; i += 10) {
        setUploadProgress(i);
        await new Promise(resolve => setTimeout(resolve, 200));
      }
      setUploading(false);
      setUploadDialog(false);
      setSnackbar({ open: true, message: 'Inspection completed successfully!', severity: 'success' });
      setSelectedFiles([]);
      setUploadProgress(0);
    };

    return (
      <Dialog open={uploadDialog} onClose={() => setUploadDialog(false)} maxWidth="md" fullWidth>
        <DialogTitle>Upload Product Images for Inspection</DialogTitle>
        <DialogContent>
          <Box sx={{ p: 3, textAlign: 'center', border: '2px dashed #ccc', borderRadius: 2, mb: 3 }}>
            <CloudUpload sx={{ fontSize: 48, color: 'text.secondary', mb: 2 }} />
            <Typography variant="h6" gutterBottom>
              Drop images here or click to browse
            </Typography>
            <input
              type="file"
              multiple
              accept="image/*"
              onChange={handleFileSelect}
              style={{ display: 'none' }}
              id="file-upload"
            />
            <label htmlFor="file-upload">
              <Button variant="outlined" component="span" startIcon={<Upload />}>
                Select Images
              </Button>
            </label>
          </Box>
          
          {selectedFiles.length > 0 && (
            <Box>
              <Typography variant="subtitle1" gutterBottom>
                Selected Files ({selectedFiles.length}):
              </Typography>
              {selectedFiles.map((file, index) => (
                <Chip key={index} label={file.name} sx={{ mr: 1, mb: 1 }} />
              ))}
            </Box>
          )}

          <FormControl fullWidth sx={{ mt: 2 }}>
            <InputLabel>Product Line</InputLabel>
            <Select value={selectedProductLine} label="Product Line">
              <MenuItem value="automotive">Automotive Line A</MenuItem>
              <MenuItem value="electronics">Electronics Line B</MenuItem>
              <MenuItem value="packaging">Packaging Line C</MenuItem>
            </Select>
          </FormControl>

          {uploading && (
            <Box sx={{ mt: 3 }}>
              <Typography variant="body2" gutterBottom>
                Processing images... {uploadProgress}%
              </Typography>
              <LinearProgress variant="determinate" value={uploadProgress} />
            </Box>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setUploadDialog(false)}>Cancel</Button>
          <Button 
            onClick={handleUpload} 
            variant="contained" 
            disabled={selectedFiles.length === 0 || uploading}
            startIcon={<PhotoCamera />}
          >
            Start Inspection
          </Button>
        </DialogActions>
      </Dialog>
    );
  };

  // Real-time monitoring component
  const RealTimeMonitor = () => {
    const [realTimeData, setRealTimeData] = useState({
      current_quality: 0.89,
      inspections_today: 156,
      active_cameras: 3,
      processing_queue: 2
    });

    useEffect(() => {
      if (realTimeMode) {
        const interval = setInterval(() => {
          setRealTimeData(prev => ({
            ...prev,
            current_quality: 0.8 + Math.random() * 0.2,
            inspections_today: prev.inspections_today + Math.floor(Math.random() * 3),
            processing_queue: Math.floor(Math.random() * 5)
          }));
        }, 2000);
        return () => clearInterval(interval);
      }
    }, [realTimeMode]);

    return (
      <Card>
        <CardContent>
          <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
            <Typography variant="h6">Real-time Monitoring</Typography>
            <Box display="flex" alignItems="center">
              <FormControlLabel
                control={
                  <Switch 
                    checked={realTimeMode} 
                    onChange={(e) => setRealTimeMode(e.target.checked)}
                    color="primary"
                  />
                }
                label="Live Mode"
              />
              <Box display="flex" alignItems="center" ml={2}>
                <Box 
                  width={12} 
                  height={12} 
                  bgcolor={realTimeMode ? 'success.main' : 'grey.400'} 
                  borderRadius="50%" 
                  mr={1}
                />
                <Typography variant="body2">
                  {realTimeMode ? 'LIVE' : 'OFFLINE'}
                </Typography>
              </Box>
            </Box>
          </Box>

          <Grid container spacing={2}>
            <Grid item xs={3}>
              <Box textAlign="center" p={2} bgcolor="primary.light" borderRadius={2}>
                <Typography variant="h4" color="primary.contrastText">
                  {(realTimeData.current_quality * 100).toFixed(1)}%
                </Typography>
                <Typography variant="body2" color="primary.contrastText">
                  Current Quality
                </Typography>
              </Box>
            </Grid>
            <Grid item xs={3}>
              <Box textAlign="center" p={2} bgcolor="success.light" borderRadius={2}>
                <Typography variant="h4" color="success.contrastText">
                  {realTimeData.inspections_today}
                </Typography>
                <Typography variant="body2" color="success.contrastText">
                  Today's Inspections
                </Typography>
              </Box>
            </Grid>
            <Grid item xs={3}>
              <Box textAlign="center" p={2} bgcolor="info.light" borderRadius={2}>
                <Typography variant="h4" color="info.contrastText">
                  {realTimeData.active_cameras}
                </Typography>
                <Typography variant="body2" color="info.contrastText">
                  Active Cameras
                </Typography>
              </Box>
            </Grid>
            <Grid item xs={3}>
              <Box textAlign="center" p={2} bgcolor="warning.light" borderRadius={2}>
                <Typography variant="h4" color="warning.contrastText">
                  {realTimeData.processing_queue}
                </Typography>
                <Typography variant="body2" color="warning.contrastText">
                  Queue Length
                </Typography>
              </Box>
            </Grid>
          </Grid>
        </CardContent>
      </Card>
    );
  };

  // System performance component
  const SystemPerformance = () => {
    const performanceData = [
      { name: 'CPU Usage', value: 23, color: '#4caf50' },
      { name: 'Memory', value: 45, color: '#2196f3' },
      { name: 'GPU', value: 67, color: '#ff9800' },
      { name: 'Storage', value: 34, color: '#9c27b0' }
    ];

    return (
      <Card>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            System Performance
          </Typography>
          <Grid container spacing={2}>
            {performanceData.map((metric) => (
              <Grid item xs={6} key={metric.name}>
                <Box mb={1}>
                  <Box display="flex" justifyContent="space-between">
                    <Typography variant="body2">{metric.name}</Typography>
                    <Typography variant="body2">{metric.value}%</Typography>
                  </Box>
                  <LinearProgress 
                    variant="determinate" 
                    value={metric.value}
                    sx={{ 
                      height: 8, 
                      borderRadius: 4,
                      backgroundColor: 'grey.200',
                      '& .MuiLinearProgress-bar': {
                        backgroundColor: metric.color
                      }
                    }}
                  />
                </Box>
              </Grid>
            ))}
          </Grid>
        </CardContent>
      </Card>
    );
  };

  // Navigation drawer
  const drawerItems = [
    { text: 'Dashboard', icon: <Dashboard />, value: 0 },
    { text: 'Inspections', icon: <PhotoCamera />, value: 1 },
    { text: 'Analytics', icon: <Analytics />, value: 2 },
    { text: 'Product Lines', icon: <Build />, value: 3 },
    { text: 'Settings', icon: <Settings />, value: 4 }
  ];

  const NavigationDrawer = () => (
    <Drawer
      variant="permanent"
      sx={{
        width: 240,
        flexShrink: 0,
        '& .MuiDrawer-paper': {
          width: 240,
          boxSizing: 'border-box',
          backgroundColor: '#1e293b',
          color: 'white'
        },
      }}
    >
      <Toolbar />
      <Box sx={{ overflow: 'auto' }}>
        <List>
          {drawerItems.map((item) => (
            <ListItem 
              button 
              key={item.text}
              onClick={() => setSelectedTab(item.value)}
              sx={{ 
                backgroundColor: selectedTab === item.value ? 'rgba(255,255,255,0.1)' : 'transparent',
                '&:hover': { backgroundColor: 'rgba(255,255,255,0.05)' }
              }}
            >
              <ListItemIcon sx={{ color: 'white' }}>{item.icon}</ListItemIcon>
              <ListItemText primary={item.text} />
            </ListItem>
          ))}
        </List>
      </Box>
    </Drawer>
  );

  // Main content based on selected tab
  const renderContent = () => {
    switch (selectedTab) {
      case 0: // Dashboard
        return (
          <Grid container spacing={3}>
            {/* Stats Cards */}
            <Grid item xs={12} sm={6} md={3}>
              <StatsCard
                title="Total Inspections"
                value={analytics.summary.total_inspections.toLocaleString()}
                subtitle="Last 30 days"
                icon={<PhotoCamera />}
                color="primary"
              />
            </Grid>
            <Grid item xs={12} sm={6} md={3}>
              <StatsCard
                title="Avg Quality Score"
                value={`${(analytics.summary.avg_quality_score * 100).toFixed(1)}%`}
                subtitle="Above target"
                icon={<TrendingUp />}
                color="success"
              />
            </Grid>
            <Grid item xs={12} sm={6} md={3}>
              <StatsCard
                title="Defects Found"
                value={analytics.summary.total_defects}
                subtitle={`${(analytics.summary.defect_rate * 100).toFixed(1)}% rate`}
                icon={<Warning />}
                color="warning"
              />
            </Grid>
            <Grid item xs={12} sm={6} md={3}>
              <StatsCard
                title="Avg Processing"
                value={`${analytics.summary.avg_processing_time}s`}
                subtitle="Per inspection"
                icon={<Speed />}
                color="info"
              />
            </Grid>

            {/* Real-time Monitor */}
            <Grid item xs={12}>
              <RealTimeMonitor />
            </Grid>

            {/* Charts */}
            <Grid item xs={12} md={8}>
              <QualityTrendsChart />
            </Grid>
            <Grid item xs={12} md={4}>
              <DefectBreakdownChart />
            </Grid>

            {/* System Performance */}
            <Grid item xs={12} md={6}>
              <SystemPerformance />
            </Grid>

            {/* Recent Inspections */}
            <Grid item xs={12} md={6}>
              <Card sx={{ height: 400 }}>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Quick Actions
                  </Typography>
                  <Grid container spacing={2}>
                    <Grid item xs={12}>
                      <Button
                        fullWidth
                        variant="contained"
                        startIcon={<PhotoCamera />}
                        onClick={() => setUploadDialog(true)}
                        sx={{ mb: 2 }}
                      >
                        New Inspection
                      </Button>
                    </Grid>
                    <Grid item xs={6}>
                      <Button
                        fullWidth
                        variant="outlined"
                        startIcon={<Analytics />}
                        onClick={() => setSelectedTab(2)}
                      >
                        View Analytics
                      </Button>
                    </Grid>
                    <Grid item xs={6}>
                      <Button
                        fullWidth
                        variant="outlined"
                        startIcon={<FileDownload />}
                      >
                        Export Report
                      </Button>
                    </Grid>
                    <Grid item xs={12}>
                      <Alert severity="info" sx={{ mt: 2 }}>
                        System operating normally. All production lines active.
                      </Alert>
                    </Grid>
                  </Grid>
                </CardContent>
              </Card>
            </Grid>
          </Grid>
        );

      case 1: // Inspections
        return (
          <Grid container spacing={3}>
            <Grid item xs={12}>
              <Box display="flex" justifyContent="between" alignItems="center" mb={3}>
                <Typography variant="h4">Inspection History</Typography>
                <Box>
                  <FormControl sx={{ mr: 2, minWidth: 120 }}>
                    <InputLabel>Product Line</InputLabel>
                    <Select
                      value={selectedProductLine}
                      label="Product Line"
                      onChange={(e) => setSelectedProductLine(e.target.value)}
                    >
                      <MenuItem value="all">All Lines</MenuItem>
                      <MenuItem value="automotive">Automotive</MenuItem>
                      <MenuItem value="electronics">Electronics</MenuItem>
                      <MenuItem value="packaging">Packaging</MenuItem>
                    </Select>
                  </FormControl>
                  <Button
                    variant="contained"
                    startIcon={<PhotoCamera />}
                    onClick={() => setUploadDialog(true)}
                  >
                    New Inspection
                  </Button>
                </Box>
              </Box>
            </Grid>
            <Grid item xs={12}>
              <InspectionsTable />
            </Grid>
          </Grid>
        );

      case 2: // Analytics
        return (
          <Grid container spacing={3}>
            <Grid item xs={12}>
              <Typography variant="h4" gutterBottom>Quality Analytics</Typography>
            </Grid>
            
            {/* Performance Metrics */}
            <Grid item xs={12} md={6}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Processing Performance
                  </Typography>
                  <ResponsiveContainer width="100%" height={300}>
                    <AreaChart data={analytics.daily_trends}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="date" />
                      <YAxis />
                      <Tooltip />
                      <Area type="monotone" dataKey="inspections" stroke="#8884d8" fill="#8884d8" />
                    </AreaChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>
            </Grid>

            <Grid item xs={12} md={6}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Defect Confidence Levels
                  </Typography>
                  <ResponsiveContainer width="100%" height={300}>
                    <BarChart data={analytics.defect_breakdown}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="type" />
                      <YAxis />
                      <Tooltip />
                      <Bar dataKey="avg_confidence" fill="#82ca9d" />
                    </BarChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>
            </Grid>

            {/* Quality Distribution */}
            <Grid item xs={12}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Quality Score Distribution
                  </Typography>
                  <Box height={300}>
                    <ResponsiveContainer width="100%" height="100%">
                      <AreaChart data={[
                        { range: '0-60%', count: 12, color: '#f44336' },
                        { range: '60-80%', count: 45, color: '#ff9800' },
                        { range: '80-90%', count: 234, color: '#2196f3' },
                        { range: '90-100%', count: 956, color: '#4caf50' }
                      ]}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="range" />
                        <YAxis />
                        <Tooltip />
                        <Area type="monotone" dataKey="count" stroke="#4caf50" fill="#4caf50" />
                      </AreaChart>
                    </ResponsiveContainer>
                  </Box>
                </CardContent>
              </Card>
            </Grid>
          </Grid>
        );

      case 3: // Product Lines
        return (
          <Grid container spacing={3}>
            <Grid item xs={12}>
              <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
                <Typography variant="h4">Product Lines</Typography>
                <Button variant="contained" startIcon={<Build />}>
                  Add Product Line
                </Button>
              </Box>
            </Grid>
            {productLines.map((line) => (
              <Grid item xs={12} md={6} key={line.id}>
                <Card>
                  <CardContent>
                    <Box display="flex" justifyContent="space-between" alignItems="center">
                      <Box>
                        <Typography variant="h6">{line.name}</Typography>
                        <Typography color="textSecondary" gutterBottom>
                          Quality Threshold: {(line.quality_threshold * 100).toFixed(0)}%
                        </Typography>
                        <Chip 
                          label={line.active ? 'Active' : 'Inactive'}
                          color={line.active ? 'success' : 'default'}
                          size="small"
                        />
                      </Box>
                      <IconButton>
                        <MoreVert />
                      </IconButton>
                    </Box>
                    <Box mt={2}>
                      <LinearProgress 
                        variant="determinate" 
                        value={line.quality_threshold * 100}
                        sx={{ height: 8, borderRadius: 4 }}
                      />
                    </Box>
                  </CardContent>
                </Card>
              </Grid>
            ))}
          </Grid>
        );

      case 4: // Settings
        return (
          <Grid container spacing={3}>
            <Grid item xs={12}>
              <Typography variant="h4" gutterBottom>System Settings</Typography>
            </Grid>
            
            {/* AI Models Status */}
            <Grid item xs={12} md={6}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    AI Models Status
                  </Typography>
                  <Box mb={2}>
                    <Box display="flex" justifyContent="space-between" alignItems="center">
                      <Typography>Defect Detection (YOLOv8)</Typography>
                      <Chip label="Active" color="success" size="small" />
                    </Box>
                    <Box display="flex" alignItems="center" mt={1}>
                      <Box flexGrow={1} mr={2}>
                        <LinearProgress variant="determinate" value={94} color="success" />
                      </Box>
                      <Typography variant="body2">94% Accuracy</Typography>
                    </Box>
                  </Box>
                  
                  <Box mb={2}>
                    <Box display="flex" justifyContent="space-between" alignItems="center">
                      <Typography>Pose Tracking (MediaPipe)</Typography>
                      <Chip label="Active" color="success" size="small" />
                    </Box>
                    <Box display="flex" alignItems="center" mt={1}>
                      <Box flexGrow={1} mr={2}>
                        <LinearProgress variant="determinate" value={91} color="info" />
                      </Box>
                      <Typography variant="body2">91% Accuracy</Typography>
                    </Box>
                  </Box>

                  <Box>
                    <Box display="flex" justifyContent="space-between" alignItems="center">
                      <Typography>Anomaly Detection</Typography>
                      <Chip label="Active" color="success" size="small" />
                    </Box>
                    <Box display="flex" alignItems="center" mt={1}>
                      <Box flexGrow={1} mr={2}>
                        <LinearProgress variant="determinate" value={87} color="warning" />
                      </Box>
                      <Typography variant="body2">87% Accuracy</Typography>
                    </Box>
                  </Box>
                </CardContent>
              </Card>
            </Grid>

            {/* System Configuration */}
            <Grid item xs={12} md={6}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    System Configuration
                  </Typography>
                  <Box mb={2}>
                    <FormControlLabel
                      control={<Switch defaultChecked />}
                      label="Real-time Processing"
                    />
                  </Box>
                  <Box mb={2}>
                    <FormControlLabel
                      control={<Switch defaultChecked />}
                      label="Pose Tracking"
                    />
                  </Box>
                  <Box mb={2}>
                    <FormControlLabel
                      control={<Switch defaultChecked />}
                      label="Anomaly Detection"
                    />
                  </Box>
                  <Box mb={2}>
                    <FormControlLabel
                      control={<Switch />}
                      label="Maintenance Mode"
                    />
                  </Box>
                  <Divider sx={{ my: 2 }} />
                  <Typography variant="subtitle2" gutterBottom>
                    Processing Settings
                  </Typography>
                  <TextField
                    fullWidth
                    label="Quality Threshold"
                    defaultValue="0.85"
                    size="small"
                    sx={{ mb: 2 }}
                  />
                  <TextField
                    fullWidth
                    label="Batch Size"
                    defaultValue="16"
                    size="small"
                  />
                </CardContent>
              </Card>
            </Grid>

            {/* Integration Status */}
            <Grid item xs={12}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Industrial Integrations
                  </Typography>
                  <Grid container spacing={2}>
                    <Grid item xs={12} md={4}>
                      <Box p={2} border={1} borderColor="grey.300" borderRadius={2}>
                        <Box display="flex" alignItems="center" mb={1}>
                          <Memory sx={{ mr: 1 }} />
                          <Typography variant="subtitle1">PLC Integration</Typography>
                          <Chip label="Connected" color="success" size="small" sx={{ ml: 'auto' }} />
                        </Box>
                        <Typography variant="body2" color="textSecondary">
                          Modbus TCP: 192.168.1.100:502
                        </Typography>
                        <Typography variant="body2" color="textSecondary">
                          Last heartbeat: 2 seconds ago
                        </Typography>
                      </Box>
                    </Grid>
                    
                    <Grid item xs={12} md={4}>
                      <Box p={2} border={1} borderColor="grey.300" borderRadius={2}>
                        <Box display="flex" alignItems="center" mb={1}>
                          <Storage sx={{ mr: 1 }} />
                          <Typography variant="subtitle1">ERP System</Typography>
                          <Chip label="Syncing" color="info" size="small" sx={{ ml: 'auto' }} />
                        </Box>
                        <Typography variant="body2" color="textSecondary">
                          SAP Integration Active
                        </Typography>
                        <Typography variant="body2" color="textSecondary">
                          Last sync: 15 minutes ago
                        </Typography>
                      </Box>
                    </Grid>
                    
                    <Grid item xs={12} md={4}>
                      <Box p={2} border={1} borderColor="grey.300" borderRadius={2}>
                        <Box display="flex" alignItems="center" mb={1}>
                          <Security sx={{ mr: 1 }} />
                          <Typography variant="subtitle1">Security</Typography>
                          <Chip label="Secure" color="success" size="small" sx={{ ml: 'auto' }} />
                        </Box>
                        <Typography variant="body2" color="textSecondary">
                          JWT Authentication
                        </Typography>
                        <Typography variant="body2" color="textSecondary">
                          SSL/TLS Enabled
                        </Typography>
                      </Box>
                    </Grid>
                  </Grid>
                </CardContent>
              </Card>
            </Grid>
          </Grid>
        );

      default:
        return (
          <Grid container spacing={3}>
            <Grid item xs={12}>
              <Card>
                <CardContent>
                  <Typography variant="h5" gutterBottom>
                    Coming Soon
                  </Typography>
                  <Typography>
                    This section is under development. Check back soon for updates!
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
          </Grid>
        );
    }
  };

  // Alerts and notifications
  const AlertsComponent = () => {
    const [alerts] = useState([
      { id: 1, type: 'warning', message: 'Quality score below threshold on Line A', time: '2 minutes ago' },
      { id: 2, type: 'info', message: 'Scheduled maintenance completed on Camera 3', time: '1 hour ago' }
    ]);

    return (
      <Menu
        anchorEl={null}
        open={false}
        onClose={() => {}}
      >
        {alerts.map((alert) => (
          <MenuItem key={alert.id}>
            <Box>
              <Typography variant="body2">{alert.message}</Typography>
              <Typography variant="caption" color="textSecondary">
                {alert.time}
              </Typography>
            </Box>
          </MenuItem>
        ))}
      </Menu>
    );
  };

  return (
    <Box sx={{ display: 'flex', minHeight: '100vh', backgroundColor: '#f5f5f5' }}>
      {/* App Bar */}
      <AppBar 
        position="fixed" 
        sx={{ 
          zIndex: (theme) => theme.zIndex.drawer + 1,
          backgroundColor: '#1976d2'
        }}
      >
        <Toolbar>
          <Typography variant="h6" noWrap component="div" sx={{ flexGrow: 1 }}>
            🔍 VisionGuard Pro - Manufacturing Quality Control
          </Typography>
          
          <Box display="flex" alignItems="center">
            <IconButton color="inherit">
              <Badge badgeContent={2} color="error">
                <Notifications />
              </Badge>
            </IconButton>
            
            <Box ml={2} display="flex" alignItems="center">
              <Avatar sx={{ width: 32, height: 32, bgcolor: 'secondary.main' }}>
                A
              </Avatar>
              <Typography variant="body2" sx={{ ml: 1 }}>
                Admin User
              </Typography>
            </Box>
          </Box>
        </Toolbar>
      </AppBar>

      {/* Navigation Drawer */}
      <NavigationDrawer />

      {/* Main Content */}
      <Box
        component="main"
        sx={{
          flexGrow: 1,
          p: 3,
          width: { sm: `calc(100% - 240px)` },
          ml: { sm: '240px' }
        }}
      >
        <Toolbar />
        
        {/* Status Banner */}
        <Box mb={3}>
          <Alert 
            severity="success" 
            action={
              <Button color="inherit" size="small" startIcon={<Refresh />}>
                Refresh
              </Button>
            }
          >
            All systems operational - 3 production lines active, 0 critical alerts
          </Alert>
        </Box>

        {/* Tab Navigation */}
        <Box mb={3}>
          <Tabs 
            value={selectedTab} 
            onChange={(e, newValue) => setSelectedTab(newValue)}
            variant="scrollable"
            scrollButtons="auto"
          >
            <Tab label="Dashboard" icon={<Dashboard />} />
            <Tab label="Inspections" icon={<PhotoCamera />} />
            <Tab label="Analytics" icon={<Analytics />} />
            <Tab label="Product Lines" icon={<Build />} />
            <Tab label="Settings" icon={<Settings />} />
          </Tabs>
        </Box>

        {/* Content */}
        {loading ? (
          <Box display="flex" justifyContent="center" alignItems="center" height="400px">
            <CircularProgress size={60} />
          </Box>
        ) : (
          renderContent()
        )}

        {/* Upload Dialog */}
        <UploadDialog />

        {/* Snackbar for notifications */}
        <Snackbar
          open={snackbar.open}
          autoHideDuration={6000}
          onClose={() => setSnackbar({ ...snackbar, open: false })}
        >
          <Alert 
            onClose={() => setSnackbar({ ...snackbar, open: false })} 
            severity={snackbar.severity}
          >
            {snackbar.message}
          </Alert>
        </Snackbar>

        {/* Floating Action Buttons */}
        <Box
          position="fixed"
          bottom={20}
          right={20}
          display="flex"
          flexDirection="column"
          gap={1}
        >
          <Button
            variant="contained"
            color="primary"
            startIcon={<PhotoCamera />}
            onClick={() => setUploadDialog(true)}
            sx={{
              borderRadius: 3,
              px: 3,
              py: 1.5,
              boxShadow: 3,
              '&:hover': { boxShadow: 6 }
            }}
          >
            Quick Inspect
          </Button>
          
          <Button
            variant="contained"
            color={realTimeMode ? "error" : "success"}
            startIcon={realTimeMode ? <Stop /> : <PlayArrow />}
            onClick={() => setRealTimeMode(!realTimeMode)}
            sx={{
              borderRadius: 3,
              px: 3,
              py: 1.5,
              boxShadow: 3,
              '&:hover': { boxShadow: 6 }
            }}
          >
            {realTimeMode ? 'Stop Live' : 'Start Live'}
          </Button>
        </Box>
      </Box>
    </Box>
  );
};

// Quality Score Gauge Component
const QualityGauge = ({ score, size = 120 }) => {
  const normalizedScore = Math.max(0, Math.min(100, score * 100));
  const strokeDasharray = 2 * Math.PI * 45; // Circumference
  const strokeDashoffset = strokeDasharray - (strokeDasharray * normalizedScore) / 100;
  
  const getColor = () => {
    if (normalizedScore >= 90) return '#4caf50';
    if (normalizedScore >= 80) return '#ff9800';
    return '#f44336';
  };

  return (
    <Box position="relative" display="inline-flex" alignItems="center" justifyContent="center">
      <svg width={size} height={size}>
        <circle
          cx={size / 2}
          cy={size / 2}
          r={45}
          stroke="#e0e0e0"
          strokeWidth="8"
          fill="none"
        />
        <circle
          cx={size / 2}
          cy={size / 2}
          r={45}
          stroke={getColor()}
          strokeWidth="8"
          fill="none"
          strokeDasharray={strokeDasharray}
          strokeDashoffset={strokeDashoffset}
          strokeLinecap="round"
          transform={`rotate(-90 ${size / 2} ${size / 2})`}
        />
      </svg>
      <Box
        position="absolute"
        display="flex"
        flexDirection="column"
        alignItems="center"
        justifyContent="center"
      >
        <Typography variant="h6" component="div" color={getColor()}>
          {normalizedScore.toFixed(0)}%
        </Typography>
        <Typography variant="caption" color="textSecondary">
          Quality
        </Typography>
      </Box>
    </Box>
  );
};

// Inspection Detail Modal
const InspectionDetailModal = ({ open, onClose, inspection }) => {
  if (!inspection) return null;

  const mockDefects = [
    {
      id: 'def-001',
      type: 'scratch',
      confidence: 0.92,
      bbox: { x: 100, y: 150, width: 50, height: 30 },
      severity: 'medium'
    },
    {
      id: 'def-002', 
      type: 'dent',
      confidence: 0.87,
      bbox: { x: 200, y: 100, width: 40, height: 25 },
      severity: 'low'
    }
  ];

  return (
    <Dialog open={open} onClose={onClose} maxWidth="lg" fullWidth>
      <DialogTitle>
        <Box display="flex" alignItems="center" justifyContent="space-between">
          <Typography variant="h6">Inspection Details</Typography>
          <QualityGauge score={inspection.quality_score} size={80} />
        </Box>
      </DialogTitle>
      <DialogContent>
        <Grid container spacing={3}>
          {/* Basic Info */}
          <Grid item xs={12} md={6}>
            <Card variant="outlined">
              <CardContent>
                <Typography variant="h6" gutterBottom>Basic Information</Typography>
                <Box mb={1}>
                  <Typography variant="body2" color="textSecondary">Inspection ID</Typography>
                  <Typography variant="body1">{inspection.id}</Typography>
                </Box>
                <Box mb={1}>
                  <Typography variant="body2" color="textSecondary">Product Line</Typography>
                  <Typography variant="body1">{inspection.product_line}</Typography>
                </Box>
                <Box mb={1}>
                  <Typography variant="body2" color="textSecondary">Timestamp</Typography>
                  <Typography variant="body1">{formatDate(inspection.timestamp)}</Typography>
                </Box>
                <Box mb={1}>
                  <Typography variant="body2" color="textSecondary">Processing Time</Typography>
                  <Typography variant="body1">1.23 seconds</Typography>
                </Box>
              </CardContent>
            </Card>
          </Grid>

          {/* Defects Found */}
          <Grid item xs={12} md={6}>
            <Card variant="outlined">
              <CardContent>
                <Typography variant="h6" gutterBottom>Defects Analysis</Typography>
                {mockDefects.length > 0 ? (
                  mockDefects.map((defect) => (
                    <Box key={defect.id} mb={2} p={2} bgcolor="grey.50" borderRadius={1}>
                      <Box display="flex" justifyContent="space-between" alignItems="center">
                        <Typography variant="subtitle2" textTransform="capitalize">
                          {defect.type}
                        </Typography>
                        <Chip 
                          label={defect.severity} 
                          color={defect.severity === 'high' ? 'error' : 'warning'}
                          size="small"
                        />
                      </Box>
                      <Typography variant="body2" color="textSecondary">
                        Confidence: {(defect.confidence * 100).toFixed(1)}%
                      </Typography>
                      <Typography variant="body2" color="textSecondary">
                        Location: ({defect.bbox.x}, {defect.bbox.y})
                      </Typography>
                    </Box>
                  ))
                ) : (
                  <Box textAlign="center" py={3}>
                    <CheckCircle color="success" sx={{ fontSize: 48, mb: 1 }} />
                    <Typography variant="body1">No defects detected</Typography>
                  </Box>
                )}
              </CardContent>
            </Card>
          </Grid>

          {/* Pose Analysis */}
          <Grid item xs={12}>
            <Card variant="outlined">
              <CardContent>
                <Typography variant="h6" gutterBottom>Pose Analysis</Typography>
                <Grid container spacing={2}>
                  <Grid item xs={12} md={3}>
                    <Box textAlign="center" p={2} bgcolor="primary.light" borderRadius={2}>
                      <Typography variant="h4" color="primary.contrastText">89%</Typography>
                      <Typography variant="body2" color="primary.contrastText">
                        Pose Confidence
                      </Typography>
                    </Box>
                  </Grid>
                  <Grid item xs={12} md={3}>
                    <Box textAlign="center" p={2} bgcolor="success.light" borderRadius={2}>
                      <Typography variant="h4" color="success.contrastText">Front</Typography>
                      <Typography variant="body2" color="success.contrastText">
                        Orientation
                      </Typography>
                    </Box>
                  </Grid>
                  <Grid item xs={12} md={6}>
                    <Box p={2} bgcolor="grey.50" borderRadius={2}>
                      <Typography variant="subtitle2" gutterBottom>Keypoints Detected</Typography>
                      <Box display="flex" flexWrap="wrap" gap={1}>
                        {['Head', 'Shoulders', 'Arms', 'Torso', 'Base'].map((point) => (
                          <Chip key={point} label={point} size="small" variant="outlined" />
                        ))}
                      </Box>
                    </Box>
                  </Grid>
                </Grid>
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      </DialogContent>
      <DialogActions>
        <Button onClick={onClose}>Close</Button>
        <Button variant="contained" startIcon={<FileDownload />}>
          Export Report
        </Button>
      </DialogActions>
    </Dialog>
  );
};

// Enhanced Inspections Table with detail modal
const EnhancedInspectionsTable = () => {
  const [selectedInspection, setSelectedInspection] = useState(null);
  const [detailModalOpen, setDetailModalOpen] = useState(false);

  const handleRowClick = (inspection) => {
    setSelectedInspection(inspection);
    setDetailModalOpen(true);
  };

  return (
    <>
      <Card>
        <CardContent>
          <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
            <Typography variant="h6">Recent Inspections</Typography>
            <Box>
              <IconButton>
                <Visibility />
              </IconButton>
              <IconButton>
                <FileDownload />
              </IconButton>
              <IconButton onClick={() => setLoading(true)}>
                <Refresh />
              </IconButton>
            </Box>
          </Box>
          <TableContainer>
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell>ID</TableCell>
                  <TableCell>Product Line</TableCell>
                  <TableCell>Quality Score</TableCell>
                  <TableCell>Defects</TableCell>
                  <TableCell>Timestamp</TableCell>
                  <TableCell>Status</TableCell>
                  <TableCell>Actions</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {inspections.map((inspection) => (
                  <TableRow 
                    key={inspection.id} 
                    hover 
                    sx={{ cursor: 'pointer' }}
                    onClick={() => handleRowClick(inspection)}
                  >
                    <TableCell>
                      <Typography variant="body2" fontFamily="monospace">
                        {inspection.id}
                      </Typography>
                    </TableCell>
                    <TableCell>{inspection.product_line}</TableCell>
                    <TableCell>
                      <Box display="flex" alignItems="center">
                        <QualityGauge score={inspection.quality_score} size={40} />
                        <Typography variant="body2" sx={{ ml: 1 }}>
                          {(inspection.quality_score * 100).toFixed(1)}%
                        </Typography>
                      </Box>
                    </TableCell>
                    <TableCell>
                      <Chip 
                        label={inspection.defects_detected}
                        color={inspection.defects_detected === 0 ? 'success' : 'error'}
                        size="small"
                      />
                    </TableCell>
                    <TableCell>
                      <Typography variant="body2">
                        {formatDate(inspection.timestamp)}
                      </Typography>
                    </TableCell>
                    <TableCell>
                      <Chip 
                        label={inspection.status}
                        color="success"
                        variant="outlined"
                        size="small"
                      />
                    </TableCell>
                    <TableCell>
                      <IconButton size="small">
                        <MoreVert />
                      </IconButton>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        </CardContent>
      </Card>

      <InspectionDetailModal
        open={detailModalOpen}
        onClose={() => setDetailModalOpen(false)}
        inspection={selectedInspection}
      />
    </>
  );
};

// Live Camera Feed Component
const LiveCameraFeed = () => {
  const [cameraActive, setCameraActive] = useState(false);

  return (
    <Card>
      <CardContent>
        <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
          <Typography variant="h6">Live Camera Feed</Typography>
          <Button
            variant={cameraActive ? "outlined" : "contained"}
            color={cameraActive ? "error" : "primary"}
            startIcon={cameraActive ? <Stop /> : <PlayArrow />}
            onClick={() => setCameraActive(!cameraActive)}
          >
            {cameraActive ? 'Stop Feed' : 'Start Feed'}
          </Button>
        </Box>
        
        <Box
          height={300}
          bgcolor="grey.100"
          borderRadius={2}
          display="flex"
          alignItems="center"
          justifyContent="center"
          border={cameraActive ? '2px solid #4caf50' : '2px dashed #ccc'}
        >
          {cameraActive ? (
            <Box textAlign="center">
              <Box 
                width={60} 
                height={60} 
                bgcolor="success.main" 
                borderRadius="50%" 
                display="flex" 
                alignItems="center" 
                justifyContent="center"
                mx="auto"
                mb={2}
              >
                <PhotoCamera sx={{ color: 'white', fontSize: 30 }} />
              </Box>
              <Typography variant="h6" color="success.main">
                Camera Feed Active
              </Typography>
              <Typography variant="body2" color="textSecondary">
                Real-time inspection in progress...
              </Typography>
            </Box>
          ) : (
            <Box textAlign="center">
              <PhotoCamera sx={{ fontSize: 48, color: 'text.secondary', mb: 2 }} />
              <Typography variant="h6" color="textSecondary">
                Camera Feed Inactive
              </Typography>
              <Typography variant="body2" color="textSecondary">
                Click "Start Feed" to begin live monitoring
              </Typography>
            </Box>
          )}
        </Box>
        
        {cameraActive && (
          <Box mt={2}>
            <Box display="flex" justifyContent="space-between" alignItems="center" mb={1}>
              <Typography variant="body2">Processing Status</Typography>
              <Chip label="Processing" color="info" size="small" />
            </Box>
            <LinearProgress />
          </Box>
        )}
      </CardContent>
    </Card>
  );
};

// Main App Component
const VisionGuardApp = () => {
  return (
    <Dashboard />
  );
};

export default VisionGuardApp;